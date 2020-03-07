#!/usr/bin/env python3
import cv2
import logging
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

from common.game_state import GameState
from common.game_state import get_wrapper_by_name
from common.replay_memory import ReplayMemory
from common.util import generate_image_for_cam_video
from common.util import grad_cam
from common.util import make_movie
from common.util import transform_h
from common.util import transform_h_inv
from common.util import visualize_cam
from sil_memory import SILReplayMemory
from termcolor import colored
from queue import Queue
from copy import deepcopy
from common_worker import CommonWorker

logger = logging.getLogger("rollout_thread")


class RolloutThread(CommonWorker):
    """Rollout Thread Class."""
    advice_confidence = 0.8
    gamma = 0.99

    def __init__(self, thread_index, action_size, env_id,
                 global_a3c, local_a3c,
                 global_pretrained_model, local_pretrained_model,
                 transformed_bellman=False, no_op_max=0):
        """Initialize A3CTrainingThread class."""

        self.is_rollout_thread = True
        self.action_size = action_size
        self.thread_idx = thread_index
        self.transformed_bellman = transformed_bellman

        self.no_op_max = no_op_max
        self.override_num_noops = 0 if self.no_op_max == 0 else None

        logger.info("===ROLLOUT thread_index: {}===".format(self.thread_idx))
        logger.info("action_size: {}".format(self.action_size))
        logger.info("reward_type: {}".format(self.reward_type))
        logger.info("transformed_bellman: {}".format(
            colored(self.transformed_bellman,
                    "green" if self.transformed_bellman else "red")))

        self.reward_clipped = True if self.reward_type == 'CLIP' else False

        # setup local a3c
        self.local_a3c = local_a3c
        self.sync_a3c = self.local_a3c.sync_from(global_a3c)

        # setup local pretrained model
        self.local_pretrained = local_pretrained_model
        self.sync_pretrained = self.local_pretrained.sync_from(global_pretrained_model)

        # setup env
        self.rolloutgame = GameState(env_id=env_id, display=False,
                            no_op_max=0, human_demo=False, episode_life=True,
                            override_num_noops=0)
        self.local_t = 0
        self.episode_reward = 0
        self.episode_steps = 0

        assert self.local_pretrained is not None
        assert self.local_a3c is not None

        self.episode = SILReplayMemory(
            self.action_size, max_len=None, gamma=self.gamma,
            clip=self.reward_clipped,
            height=self.local_a3c.in_shape[0],
            width=self.local_a3c.in_shape[1],
            phi_length=self.local_a3c.in_shape[2],
            reward_constant=self.reward_constant)


    def record_rollout(self, score=0, steps=0,
                       old_return=0, new_return=0,
                       global_t=0, rollout_ctr=0,
                       mode='Rollout', confidence=None, episodes=None):
        """Record rollout summary."""
        summary = tf.Summary()
        summary.value.add(tag='{}/score'.format(mode),
                          simple_value=float(score))
        summary.value.add(tag='{}/old_return_from_s'.format(mode),
                          simple_value=float(old_return))
        summary.value.add(tag='{}/new_return_from_s'.format(mode),
                          simple_value=float(new_return))
        summary.value.add(tag='{}/steps'.format(mode),
                          simple_value=float(steps))
        summary.value.add(tag='{}/rollout_ctr'.format(mode),
                          simple_value=float(rollout_ctr))
        if confidence is not None:
            summary.value.add(tag='{}/advice-confidence'.format(mode),
                              simple_value=float(confidence))
        if episodes is not None:
            summary.value.add(tag='{}/episodes'.format(mode),
                              simple_value=float(episodes))
        self.writer.add_summary(summary, global_t)
        self.writer.flush()

    def compute_return_for_state(self, rewards, terminal):
        """Compute expected return."""
        length = np.shape(rewards)[0]
        returns = np.empty_like(rewards, dtype=np.float32)

        if self.reward_clipped:
            rewards = np.clip(rewards, -1., 1.)
        else:
            rewards = np.sign(rewards) * self.reward_constant + rewards

        for i in reversed(range(length)):
            if terminal[i]:
                returns[i] = rewards[i] if self.reward_clipped else transform_h(rewards[i])
            else:
                if self.reward_clipped:
                    returns[i] = rewards[i] + self.gamma * returns[i+1]
                else:
                    # apply transformed expected return
                    exp_r_t = self.gamma * transform_h_inv(returns[i+1])
                    returns[i] = transform_h(rewards[i] + exp_r_t)
        return returns[0]

    def rollout(self, a3c_sess, pretrain_sess, global_t, badstate, rollout_ctr, add_all_rollout):
        """Rollout, one at a time."""
        # a3c_sess.run(self.sync_a3c)
        pretrain_sess.run(self.sync_pretrained)

        # assert pretrain_sess.run(self.local_pretrained.W_fc2).all() == \
        #     pretrain_sess.run(global_pretrained.W_fc2).all()

        # for each bad state in queue, do rollout till terminal (no max=20 limit)
        _, fs, a, old_return, _ = badstate

        rewards = []
        terminals = []
        confidences = []

        terminal_pseudo = False  # loss of life
        terminal_end = False  # real terminal
        add = False

        self.rolloutgame.reset(hard_reset=True)
        self.rolloutgame.restore_full_state(fs)
        # check if restore successful
        assert self.rolloutgame.s_t.all() == fs.all()

        start_local_t = self.local_t

        while True:
            state = cv2.resize(self.rolloutgame.s_t,
                       self.local_a3c.in_shape[:-1],
                       interpolation=cv2.INTER_AREA)
            fullstate = self.rolloutgame.clone_full_state()

            model_pi = self.local_pretrained.run_policy(pretrain_sess, state)
            action, confidence = self.choose_action_with_high_confidence(
                model_pi, exclude_noop=False)

            confidences.append(confidence)

            self.rolloutgame.step(action)

            reward = self.rolloutgame.reward
            terminal = self.rolloutgame.terminal
            terminals.append(terminal)

            self.episode_reward += reward

            self.episode.add_item(self.rolloutgame.s_t, fullstate, action,
                                  reward, terminal, from_rollout=True)

            if self.reward_type == 'LOG':
                reward = np.sign(reward) * np.log(1 + np.abs(reward))
            elif self.reward_type == 'CLIP':
                reward = np.sign(reward)
            rewards.append(reward)

            self.local_t += 1
            self.episode_steps += 1
            global_t += 1

            self.rolloutgame.update()

            if terminal:
                terminal_pseudo = True
                env = self.rolloutgame.env
                name = 'EpisodicLifeEnv'
                terminal_end = get_wrapper_by_name(env, name).was_real_done
                if rollout_ctr % 10 == 0 and rollout_ctr > 0:
                    log_msg = "ROLLOUT: rollout_ctr={} worker={} global_t={} local_t={}".format(
                        rollout_ctr, self.thread_idx, global_t, self.local_t)
                    score_str = colored("score={}".format(
                        self.episode_reward), "magenta")
                    steps_str = colored("steps={}".format(
                        self.episode_steps), "blue")
                    conf_str = colored("advice-confidence={}".format(
                        np.mean(confidences)), "blue")
                    log_msg += " {} {} {}".format(score_str, steps_str, conf_str)
                    logger.info(log_msg)

                new_return = self.compute_return_for_state(rewards, terminals)
                if not add_all_rollout:
                    if new_return > old_return:
                        add = True
                        self.record_rollout(
                            score=self.episode_reward, steps=self.episode_steps,
                            old_return=old_return, new_return=new_return,
                            global_t=global_t, rollout_ctr=rollout_ctr, mode='Rollout',
                            confidence=np.mean(confidences), episodes=None)
                        rollout_ctr += 1
                else:
                    add = True
                    self.record_rollout(
                        score=self.episode_reward, steps=self.episode_steps,
                        old_return=old_return, new_return=new_return,
                        global_t=global_t, rollout_ctr=rollout_ctr, mode='Rollout',
                        confidence=np.mean(confidences), episodes=None)
                    rollout_ctr += 1

                self.episode_reward = 0
                self.episode_steps = 0
                self.rolloutgame.reset(hard_reset=True)

                break

        diff_local_t = self.local_t - start_local_t
        return diff_local_t, terminal_end, terminal_pseudo, rollout_ctr, add
