#!/usr/bin/env python3
import cv2
import logging
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import math

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
                 global_a3c, local_a3c, update_in_rollout, nstep_bc,
                 global_pretrained_model, local_pretrained_model,
                 transformed_bellman=False, no_op_max=0,
                 device='/cpu:0', entropy_beta=0.01, clip_norm=None,
                 grad_applier=None, initial_learn_rate=0.007,
                 learning_rate_input=None):
        """Initialize RolloutThread class."""
        self.is_rollout_thread = True
        self.action_size = action_size
        self.thread_idx = thread_index
        self.transformed_bellman = transformed_bellman
        self.entropy_beta = entropy_beta
        self.clip_norm = clip_norm
        self.initial_learning_rate = initial_learn_rate
        self.learning_rate_input = learning_rate_input

        self.no_op_max = no_op_max
        self.override_num_noops = 0 if self.no_op_max == 0 else None

        logger.info("===ROLLOUT thread_index: {}===".format(self.thread_idx))
        logger.info("device: {}".format(device))
        logger.info("action_size: {}".format(self.action_size))
        logger.info("reward_type: {}".format(self.reward_type))
        logger.info("transformed_bellman: {}".format(
            colored(self.transformed_bellman,
                    "green" if self.transformed_bellman else "red")))
        logger.info("update in rollout: {}".format(
            colored(update_in_rollout, "green" if update_in_rollout else "red")))
        logger.info("N-step BC: {}".format(nstep_bc))

        self.reward_clipped = True if self.reward_type == 'CLIP' else False

        # setup local a3c
        self.local_a3c = local_a3c
        self.sync_a3c = self.local_a3c.sync_from(global_a3c)
        with tf.device(device):
            local_vars = self.local_a3c.get_vars
            self.local_a3c.prepare_loss(
                entropy_beta=self.entropy_beta, critic_lr=0.5)
            var_refs = [v._ref() for v in local_vars()]
            self.rollout_gradients = tf.gradients(self.local_a3c.total_loss, var_refs)
            global_vars = global_a3c.get_vars
            if self.clip_norm is not None:
                self.rollout_gradients, grad_norm = tf.clip_by_global_norm(
                    self.rollout_gradients, self.clip_norm)
            self.rollout_gradients = list(zip(self.rollout_gradients, global_vars()))
            self.rollout_apply_gradients = grad_applier.apply_gradients(self.rollout_gradients)

        # setup local pretrained model
        self.local_pretrained = None
        if nstep_bc > 0:
            assert local_pretrained_model is not None
            assert global_pretrained_model is not None
            self.local_pretrained = local_pretrained_model
            self.sync_pretrained = self.local_pretrained.sync_from(global_pretrained_model)

        # setup env
        self.rolloutgame = GameState(env_id=env_id, display=False,
                            no_op_max=0, human_demo=False, episode_life=True,
                            override_num_noops=0)
        self.local_t = 0
        self.episode_reward = 0
        self.episode_steps = 0

        self.action_meaning = self.rolloutgame.env.unwrapped.get_action_meanings()

        assert self.local_a3c is not None
        if nstep_bc > 0:
            assert self.local_pretrained is not None

        self.episode = SILReplayMemory(
            self.action_size, max_len=None, gamma=self.gamma,
            clip=self.reward_clipped,
            height=self.local_a3c.in_shape[0],
            width=self.local_a3c.in_shape[1],
            phi_length=self.local_a3c.in_shape[2],
            reward_constant=self.reward_constant)


    def record_rollout(self, score=0, steps=0,
                       old_return=0, new_return=0,
                       global_t=0, rollout_ctr=0, rollout_added_ctr=0,
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
        summary.value.add(tag='{}/all_rollout_ctr'.format(mode),
                          simple_value=float(rollout_ctr))
        summary.value.add(tag='{}/rollout_added_ctr'.format(mode),
                          simple_value=float(rollout_added_ctr))
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

    def update_a3c(self, sess, actions, states, rewards, values, global_t):
        cumsum_reward = 0.0
        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_state = []
        batch_action = []
        batch_adv = []
        batch_cumsum_reward = []

        # compute and accumulate gradients
        for(ai, ri, si, vi) in zip(actions, rewards, states, values):
            if self.transformed_bellman:
                ri = np.sign(ri) * self.reward_constant + ri
                cumsum_reward = transform_h(
                    ri + self.gamma * transform_h_inv(cumsum_reward))
            else:
                cumsum_reward = ri + self.gamma * cumsum_reward
            advantage = cumsum_reward - vi

            # convert action to one-hot vector
            a = np.zeros([self.action_size])
            a[ai] = 1

            batch_state.append(si)
            batch_action.append(a)
            batch_adv.append(advantage)
            batch_cumsum_reward.append(cumsum_reward)

        cur_learning_rate = self._anneal_learning_rate(global_t,
                self.initial_learning_rate )

        feed_dict = {
            self.local_a3c.s: batch_state,
            self.local_a3c.a: batch_action,
            self.local_a3c.advantage: batch_adv,
            self.local_a3c.cumulative_reward: batch_cumsum_reward,
            self.learning_rate_input: cur_learning_rate,
            }

        sess.run(self.rollout_apply_gradients, feed_dict=feed_dict)

        return batch_adv

    def rollout(self, a3c_sess, folder, pretrain_sess, game, global_t, badstate,
                rollout_ctr, rollout_added_ctr, rollout_sample_used, rollout_sample_used_adv,
                rollout_new_return, rollout_old_return,
                add_all_rollout, ep_max_steps, nstep_bc, update_in_rollout):
        """Rollout, one at a time."""
        a3c_sess.run(self.sync_a3c)
        if nstep_bc > 0:
            pretrain_sess.run(self.sync_pretrained)

        # assert pretrain_sess.run(self.local_pretrained.W_fc2).all() == \
        #     pretrain_sess.run(global_pretrained.W_fc2).all()

        # for each bad state in queue, do rollout till terminal (no max=20 limit)
        _, fs, old_a, old_return, _, _ = badstate

        states = []
        actions = []
        rewards = []
        values = []
        terminals = []
        confidences = []

        terminal_pseudo = False  # loss of life
        terminal_end = False  # real terminal
        add = False

        self.rolloutgame.reset(hard_reset=True)
        self.rolloutgame.restore_full_state(fs)
        # check if restore successful
        # assert self.rolloutgame.s_t.all() == fs.all()
        fs_check = self.rolloutgame.clone_full_state()
        assert fs_check.all() == fs.all()
        del fs_check

        start_local_t = self.local_t
        self.rolloutgame.step(0)
        # self.rolloutgame.update()

        # # record video of rollout
        # video_buffer = []
        # init_img = self.rolloutgame.get_screen_rgb()
        # video_buffer.append(init_img)

        # prevent breakout from stucking,
        # see https://github.com/openai/gym/blob/54f22cf4db2e43063093a1b15d968a57a32b6e90/gym/envs/__init__.py#L635
        while ep_max_steps > 0: #True:
            # print(ep_max_steps)
            # self.rolloutgame.env.render()
            # time.sleep(0.01)
            state = cv2.resize(self.rolloutgame.s_t,
                       self.local_a3c.in_shape[:-1],
                       interpolation=cv2.INTER_AREA)
            fullstate = self.rolloutgame.clone_full_state()

            if nstep_bc > 0:
                # print("taking action from BC, {} steps left".format(nstep_bc))
                model_pi = self.local_pretrained.run_policy(pretrain_sess, state)
                # if game == "Breakout": # breakout needs some stocasity
                #     action = self.egreedy_action(model_pi, epsilon=0.01)
                #     confidences.append(model_pi[action])
                # else:
                action, confidence = self.choose_action_with_high_confidence(
                                          model_pi, exclude_noop=False)
                confidences.append(confidence)
                nstep_bc -= 1
            else:
                # print("taking action from A3C")
                pi_, _, logits_ = self.local_a3c.run_policy_and_value(a3c_sess,
                                                                      state)
                action = self.pick_action(logits_)
                confidences.append(pi_[action])

            value_ = self.local_a3c.run_value(a3c_sess, state)
            values.append(value_)
            states.append(state)
            actions.append(action)

            self.rolloutgame.step(action)

            ep_max_steps-=1

            reward = self.rolloutgame.reward
            terminal = self.rolloutgame.terminal
            terminals.append(terminal)
            # print(reward, terminal)

            # video_buffer.append(self.rolloutgame.get_screen_rgb())

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
                rollout_ctr += 1
                terminal_end = get_wrapper_by_name(env, name).was_real_done
                if rollout_ctr % 10 == 0 and rollout_ctr > 0:
                    log_msg = "ROLLOUT: rollout_ctr={} added_rollout_ct={} worker={} global_t={} local_t={}".format(
                        rollout_ctr, rollout_added_ctr, self.thread_idx, global_t, self.local_t)
                    score_str = colored("score={}".format(
                        self.episode_reward), "magenta")
                    steps_str = colored("steps={}".format(
                        self.episode_steps), "blue")
                    conf_str = colored("advice-confidence={}".format(
                        np.mean(confidences)), "blue")
                    log_msg += " {} {} {}".format(score_str, steps_str, conf_str)
                    logger.info(log_msg)

                new_return = self.compute_return_for_state(rewards, terminals)
                # print("new&old returns: ",new_return, old_return)
                if not add_all_rollout:
                    # if game == "Pong":
                    #     # print("add old return for Pong")
                    #     old_return += 0.1
                    if new_return > old_return:
                        add = True

                else:
                    add = True

                if add:
                    rollout_added_ctr += 1
                    rollout_new_return += new_return
                    rollout_old_return += old_return
                    # update policy immediate using a good rollout
                    if update_in_rollout:
                        logger.info("Update A3C using rollout data")
                        batch_adv = self.update_a3c(a3c_sess, actions, states, rewards, values, global_t)
                        rollout_sample_used += len(actions)
                        rollout_sample_used_adv += np.sum(batch_adv)

                # if add and old_a != actions[-1]:
                #     # save the img of the rollout state, check old_action vs. new_action
                #     plt.imshow(init_img)
                #     plt.axis('off')
                #     plt.title("old: "+str(self.action_meaning[old_a])+
                #               " new: "+str(self.action_meaning[actions[-1]]))
                #     file = 'rollout/image{ep:010d}'.format(ep=global_t)
                #     plt.savefig(str(folder / file),bbox_inches='tight')
                #     # save the rollout video
                #     time_per_step = 0.0167
                #     images = np.array(video_buffer)
                #     file = 'rollout/video{ep:010d}'.format(ep=global_t)
                #     duration = len(images)*time_per_step
                #     make_movie(images, str(folder / file),
                #                duration=duration, true_image=True,
                #                salience=False)
                # del video_buffer

                self.record_rollout(
                    score=self.episode_reward, steps=self.episode_steps,
                    old_return=old_return, new_return=new_return,
                    global_t=global_t, rollout_ctr=rollout_ctr,
                    rollout_added_ctr=rollout_added_ctr,
                    mode='Rollout',
                    confidence=np.mean(confidences), episodes=None)

                self.episode_reward = 0
                self.episode_steps = 0
                self.rolloutgame.reset(hard_reset=True)
                break

        diff_local_t = self.local_t - start_local_t

        return diff_local_t, terminal_end, terminal_pseudo, rollout_ctr, \
               rollout_added_ctr, add, rollout_sample_used, rollout_sample_used_adv,\
               rollout_new_return, rollout_old_return
