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

logger = logging.getLogger("a3c_training_thread")


class RolloutThread(object):
    """Rollout Thread Class."""

    log_interval = 100
    perf_log_interval = 1000
    local_t_max = 20
    use_lstm = False
    entropy_beta = 0.01
    gamma = 0.99
    use_mnih_2015 = False
    reward_type = 'CLIP'  # CLIP | LOG | RAW
    finetune_upper_layers_only = False
    shaping_reward = 0.001
    shaping_factor = 1.
    shaping_gamma = 0.85
    advice_confidence = 0.8
    shaping_actions = -1  # -1 all actions, 0 exclude noop
    clip_norm = 0.5
    use_grad_cam = False
    use_sil = False
    use_sil_neg = False # test if also using samples that are (G-V<0)
    # use_correction = False
    log_idx = 0
    reward_constant = 0

    def __init__(self, thread_index, action_size, env_id,
                 global_a3c, local_a3c,
                 global_pretrained_model, local_pretrained_model,
                 max_global_time_step=0, device=None,
                 sil_thread=False, classify_thread=False, rollout_thread=True,
                 transformed_bellman=False,
                 no_op_max=0):
        """Initialize A3CTrainingThread class."""
        self.action_size = action_size
        self.thread_idx = thread_index
        self.max_global_time_step = max_global_time_step
        self.sil_thread = sil_thread
        if self.sil_thread:
            self.batch_size = batch_size
        self.classify_thread = classify_thread
        self.rollout_thread = rollout_thread
        self.transformed_bellman = transformed_bellman

        self.no_op_max = no_op_max
        self.override_num_noops = 0 if self.no_op_max == 0 else None

        logger.info("===ROLLOUT thread_index: {}".format(self.thread_idx))
        logger.info("device: {}".format(device))
        logger.info("sil_thread: {}".format(
            colored(self.sil_thread, "green" if self.sil_thread else "red")))
        logger.info("classifier_thread: {}".format(
            colored(self.classify_thread, "green" if self.classify_thread else "red")))
        logger.info("rollout_thread: {}".format(
            colored(self.rollout_thread, "green" if self.rollout_thread else "red")))

        logger.info("local_t_max: {}".format(self.local_t_max))
        logger.info("use_lstm: {}".format(
            colored(self.use_lstm, "green" if self.use_lstm else "red")))
        logger.info("action_size: {}".format(self.action_size))
        logger.info("entropy_beta: {}".format(self.entropy_beta))
        logger.info("gamma: {}".format(self.gamma))
        logger.info("reward_type: {}".format(self.reward_type))
        logger.info("finetune_upper_layers_only: {}".format(
            colored(self.finetune_upper_layers_only,
                    "green" if self.finetune_upper_layers_only else "red")))
        logger.info("transformed_bellman: {}".format(
            colored(self.transformed_bellman,
                    "green" if self.transformed_bellman else "red")))
        logger.info("clip_norm: {}".format(self.clip_norm))
        logger.info("use_grad_cam: {}".format(
            colored(self.use_grad_cam,
                    "green" if self.use_grad_cam else "red")))

        reward_clipped = True if self.reward_type == 'CLIP' else False

        # setup local a3c
        self.local_a3c = local_a3c
        self.sync_a3c = self.local_a3c.sync_from(global_a3c)

        # setup local pretrained model
        self.local_pretrained = local_pretrained_model
        # self.global_pretrained = global_pretrained_model
        self.sync_pretrained = self.local_pretrained.sync_from(global_pretrained_model)

        # setup env
        self.rolloutgame = GameState(env_id=env_id, display=False,
                            no_op_max=0, human_demo=False, episode_life=True,
                            override_num_noops=0)
        self.local_t = 0
        self.episode_reward = 0
        self.episode_steps = 0
        # # variable controlling log output
        # self.prev_local_t = 0

        assert self.local_pretrained is not None
        assert self.local_a3c is not None

        self.episode = SILReplayMemory(
            self.action_size, max_len=None, gamma=self.gamma,
            clip=reward_clipped,
            height=self.local_a3c.in_shape[0],
            width=self.local_a3c.in_shape[1],
            phi_length=self.local_a3c.in_shape[2],
            reward_constant=self.reward_constant)


    def pick_action(self, logits):
        """Choose action probabilistically.

        Reference:
        https://github.com/ppyht2/tf-a2c/blob/master/src/policy.py
        """
        noise = np.random.uniform(0, 1, np.shape(logits))
        return np.argmax(logits - np.log(-np.log(noise)))

    def pick_action_w_confidence(self, pi_values, exclude_noop=True):
        """Pick action with confidence."""
        max_confidence_action = np.argmax(pi_values[1 if exclude_noop else 0:])
        confidence = pi_values[max_confidence_action]
        return (max_confidence_action+(1 if exclude_noop else 0)), confidence

    def set_summary_writer(self, writer):
        """Set summary writer."""
        self.writer = writer

    def record_summary(self, score=0, steps=0, episodes=None, global_t=0,
                       mode='Test'):
        """Record summary."""
        summary = tf.Summary()
        summary.value.add(tag='{}/score'.format(mode),
                          simple_value=float(score))
        summary.value.add(tag='{}/steps'.format(mode),
                          simple_value=float(steps))
        if episodes is not None:
            summary.value.add(tag='{}/episodes'.format(mode),
                              simple_value=float(episodes))
        self.writer.add_summary(summary, global_t)
        self.writer.flush()

    def record_rollout(self, score=0, steps=0, global_t=0, rollout_ctr=0,
                          mode='Rollout', confidence=None, episodes=None):
        """Record summary."""
        summary = tf.Summary()
        summary.value.add(tag='{}/score'.format(mode),
                          simple_value=float(score))
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

    def set_start_time(self, start_time):
        """Set start time."""
        self.start_time = start_time


    def testing(self, sess, max_steps, global_t, folder, demo_memory_cam=None, worker=None):
        """Evaluate A3C."""
        assert worker is not None
        logger.info("Evaluate policy at global_t={}...".format(global_t))

        # copy weights from shared to local
        sess.run(worker.sync)

        if demo_memory_cam is not None and global_t % 5000000 == 0:
            self.generate_cam_video(sess, 0.03, global_t, folder,
                                    demo_memory_cam)

        episode_buffer = []
        worker.game_state.reset(hard_reset=True)
        episode_buffer.append(worker.game_state.get_screen_rgb())

        total_reward = 0
        total_steps = 0
        episode_reward = 0
        episode_steps = 0
        n_episodes = 0
        while max_steps > 0:
            state = cv2.resize(worker.game_state.s_t,
                               worker.local_net.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)
            pi_, value_, logits_ = worker.local_net.run_policy_and_value(
                sess, state)

            if False:
                action = np.random.choice(range(worker.action_size), p=pi_)
            else:
                action = worker.pick_action(logits_)

            # take action
            worker.game_state.step(action)
            terminal = worker.game_state.terminal

            if n_episodes == 0 and global_t % 5000000 == 0:
                episode_buffer.append(worker.game_state.get_screen_rgb())

            episode_reward += worker.game_state.reward
            episode_steps += 1
            max_steps -= 1

            # s_t = s_t1
            worker.game_state.update()

            if terminal:
                env = worker.game_state.env
                name = 'EpisodicLifeEnv'
                if get_wrapper_by_name(env, name).was_real_done:
                    if n_episodes == 0 and global_t % 5000000 == 0:
                        time_per_step = 0.0167
                        images = np.array(episode_buffer)
                        file = 'frames/image{ep:010d}'.format(ep=global_t)
                        duration = len(images)*time_per_step
                        make_movie(images, str(folder / file),
                                   duration=duration, true_image=True,
                                   salience=False)
                        episode_buffer = []
                    n_episodes += 1
                    score_str = colored("score={}".format(episode_reward),
                                        "yellow")
                    steps_str = colored("steps={}".format(episode_steps),
                                        "cyan")
                    log_data = (global_t, worker.thread_idx, self.thread_idx,
                                n_episodes, score_str, steps_str,
                                total_steps)
                    logger.debug("test: global_t={} test_worker={} cur_worker={}"
                                 " trial={} {} {}"
                                 " total_steps={}".format(*log_data))
                    total_reward += episode_reward
                    total_steps += episode_steps
                    episode_reward = 0
                    episode_steps = 0

                worker.game_state.reset(hard_reset=False)
                if worker.use_lstm:
                    worker.local_net.reset_state()

        if n_episodes == 0:
            total_reward = episode_reward
            total_steps = episode_steps
        else:
            total_reward = total_reward / n_episodes
            total_steps = total_steps // n_episodes

        log_data = (global_t, worker.thread_idx, self.thread_idx,
                    total_reward, total_steps,
                    n_episodes)
        logger.info("test: global_t={} test_worker={} cur_worker={}"
                    " final score={} final steps={}"
                    " # trials={}".format(*log_data))

        worker.record_summary(
            score=total_reward, steps=total_steps,
            episodes=n_episodes, global_t=global_t, mode='A3C_Test')

        # reset variables used in training
        worker.episode_reward = 0
        worker.episode_steps = 0
        worker.game_state.reset(hard_reset=True)
        worker.last_rho = 0.

        if self.use_lstm:
            worker.local_net.reset_state()

        if worker.use_sil and not worker.sil_thread:
            # ensure no states left from a non-terminating episode
            worker.episode.reset()
        return (total_reward, total_steps, n_episodes)



    def test_game_classifier(self, global_t, max_steps, sess, worker=None):
        """Evaluate game with current classifier model."""
        assert worker is not None
        logger.info("Testing classifier at global_t={}...".format(global_t))

        worker.game_state.reset(hard_reset=True)

        # max_steps = 10000
        total_reward = 0
        total_steps = 0
        episode_reward = 0
        episode_steps = 0
        n_episodes = 0
        while max_steps > 0:
            state = cv2.resize(worker.game_state.s_t,
                               worker.net.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)
            model_pi = worker.net.run_policy(sess, state)
            action, confidence = worker.choose_action_with_high_confidence(
                model_pi, exclude_noop=True)

            # take action
            worker.game_state.step(action)
            terminal = worker.game_state.terminal
            episode_reward += worker.game_state.reward
            episode_steps += 1
            max_steps -= 1

            # s_t = s_t1
            worker.game_state.update()

            if terminal:
                was_real_done = get_wrapper_by_name(
                    worker.game_state.env, 'EpisodicLifeEnv').was_real_done

                if was_real_done:
                    n_episodes += 1
                    score_str = colored("score={}".format(
                        episode_reward), "magenta")
                    steps_str = colored("steps={}".format(
                        episode_steps), "blue")
                    log_data = (n_episodes, score_str, steps_str,
                                worker.thread_idx, self.thread_idx, total_steps)
                    logger.debug("classifier test: trial={} {} {} "
                                 "test_worker={} cur_worker={} total_steps={}"
                                 .format(*log_data))
                    total_reward += episode_reward
                    total_steps += episode_steps
                    episode_reward = 0
                    episode_steps = 0

                worker.game_state.reset(hard_reset=False)

        if n_episodes == 0:
            total_reward = episode_reward
            total_steps = episode_steps
        else:
            total_reward = total_reward / n_episodes
            total_steps = total_steps // n_episodes

        log_data = (global_t, worker.thread_idx, self.thread_idx,
                    total_reward, total_steps, n_episodes)
        logger.info("classifier test: global_t={} test_worker={} cur_worker={} "
                    "final score={} final steps={} # trials={}"
                    .format(*log_data))
        self.record_summary(
            score=total_reward, steps=total_steps,
            episodes=n_episodes, global_t=global_t, mode='Classifier_Test')


    def rollout(self, a3c_sess, pretrain_sess, global_t, badstate_queue, rollout_ctr):
        """Rollout, one at a time."""
        a3c_sess.run(self.sync_a3c)
        pretrain_sess.run(self.sync_pretrained)

        # assert pretrain_sess.run(self.local_pretrained.W_fc2).all() == \
        #     pretrain_sess.run(global_pretrained.W_fc2).all()

        # for each bad state in queue, do rollout till terminal (no max=20 limit)
        data = badstate_queue.get()
        s, fs, a, _return = data

        # states = []
        # fullstates = []
        # actions = []
        rewards = []
        confidences = []

        terminal_pseudo = False  # loss of life
        terminal_end = False  # real terminal

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
            action, confidence = self.pick_action_w_confidence(
                model_pi, exclude_noop=False)

            # assert action != 0

            # states.append(state)
            # fullstates.append(fullstate)
            # actions.append(action)
            confidences.append(confidence)

            self.rolloutgame.step(action)

            reward = self.rolloutgame.reward
            terminal = self.rolloutgame.terminal

            self.episode_reward += reward

            self.episode.add_item(self.rolloutgame.s_t, fullstate, action,
                                  reward, terminal)

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
                if rollout_ctr % 100 == 0 and rollout_ctr > 0:
                    log_msg = "rollout: rollout_ctr={} worker={} global_t={} local_t={}".format(
                        rollout_ctr, self.thread_idx, global_t, self.local_t)
                    score_str = colored("score={}".format(
                        self.episode_reward), "magenta")
                    steps_str = colored("steps={}".format(
                        self.episode_steps), "blue")
                    conf_str = colored("advice-confidence={}".format(
                        np.mean(confidences)), "blue")
                    log_msg += " {} {} {}".format(score_str, steps_str, conf_str)
                    logger.info(log_msg)
                # train_rewards['train'][global_t] = (self.episode_reward,
                #                                     self.episode_steps)
                self.record_rollout(
                    score=self.episode_reward, steps=self.episode_steps,
                    global_t=global_t, rollout_ctr=rollout_ctr, mode='Rollout',
                    confidence=np.mean(confidences), episodes=None)
                self.episode_reward = 0
                self.episode_steps = 0
                self.rolloutgame.reset(hard_reset=True)
                rollout_ctr += 1
                break

        diff_local_t = self.local_t - start_local_t
        return badstate_queue, diff_local_t, terminal_end, terminal_pseudo, rollout_ctr
