#!/usr/bin/env python3
"""Supervised learning as pre-training method.

This module uses supervised learning as a pre-training method to learn features
using data from human demonstrations. There are two training options:
1) regular multi-class classification, and 2) one class vs. all class using
multi-task learning technique of having multiple output layers for each class.

Usage:
    Multi-class classification
        $ python3 pretrain/run_experiment.py
            --gym-env=PongNoFrameskip-v4
            --classify-demo --use-mnih-2015
            --train-max-steps=150000 --batch_size=32
"""
import cv2
import logging
import numpy as np
import os
import pathlib
import signal
import sys

from common.game_state import GameState
from common.game_state import get_wrapper_by_name
from common.replay_memory import ReplayMemoryReturns
from common.util import load_memory
from common.util import LogFormatter
from common.util import percent_decrease
from common.util import solve_weight
from class_network import MultiClassNetwork
from termcolor import colored
from copy import deepcopy
from common_worker import CommonWorker

logger = logging.getLogger("classify_demo")


class ClassifyDemo(CommonWorker):
    """Use Supervised learning for learning features."""

    def __init__(self, tf, net, thread_index, game_name, train_max_steps, batch_size,
                 ae_grad_applier=None, grad_applier=None, eval_freq=5000,
                 demo_memory_folder=None, demo_ids=None, folder=None,
                 exclude_num_demo_ep=0, device='/cpu:0', clip_norm=None, game_state=None,
                 sampling_type=None, sl_loss_weight=1.0, reward_constant=0,
                 exp_buffer=None,):
        """Initialize ClassifyDemo class."""
        assert game_state is not None

        self.is_classify_thread = True

        self.net = net
        self.name = game_name
        self.train_max_steps = train_max_steps
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.folder = folder
        self.tf = tf
        self.exclude_num_demo_ep = exclude_num_demo_ep
        self.stop_requested = False
        self.game_state = game_state
        self.best_model_reward = -(sys.maxsize)
        self.sampling_type = sampling_type
        self.thread_idx = thread_index

        logger.info("===CLASSIFIER thread_index {}".format(self.thread_idx))
        logger.info("device: {}".format(device))

        logger.info("train_max_steps: {}".format(self.train_max_steps))
        logger.info("batch_size: {}".format(self.batch_size))
        logger.info("eval_freq: {}".format(self.eval_freq))
        logger.info("sampling_type: {}".format(self.sampling_type))
        logger.info("clip_norm: {}".format(clip_norm))
        logger.info("reward_constant: {}".format(reward_constant))

        self.net.prepare_loss(sl_loss_weight=1.0, val_weight=0.5)
        self.net.prepare_evaluate()
        self.apply_gradients = self.prepare_compute_gradients(
            grad_applier, device, clip_norm=clip_norm)

        sample = exp_buffer.sample(512, beta=0.4)
        _, batch, _ = sample
        batch_state, batch_action, _, _ = batch

        self.test_batch_si = deepcopy(batch_state)

        self.test_batch_a = deepcopy(batch_action)

        del batch_state, batch_action

    def prepare_compute_gradients(self, grad_applier, device, clip_norm=None):
        """Return operation for gradient application.

        Keyword arguments:
        grad_applier -- optimizer for applying gradients
        device -- cpu or gpu
        clip_norm -- value for clip_by_global_norm (default None)
        """
        with self.net.graph.as_default():
            with self.tf.device(device):
                apply_gradients = self.__compute_gradients(
                    grad_applier, self.net.total_loss, clip_norm)

            return apply_gradients

    def __compute_gradients(self, grad_applier, total_loss, clip_norm=None):
        """Apply gradient clipping and return op for gradient application."""
        grads_vars = grad_applier.compute_gradients(total_loss)
        grads = []
        params = []
        for p in grads_vars:
            if p[0] is None:
                continue
            grads.append(p[0])
            params.append(p[1])

        if clip_norm is not None:
            grads, _ = self.tf.clip_by_global_norm(grads, clip_norm)

        grads_vars_updates = zip(grads, params)
        return grad_applier.apply_gradients(grads_vars_updates)

    def save_best_model(self, test_reward, best_saver, sess):
        """Save best network model's parameters and reward to file.

        Keyword arguments:
        test_reward -- testing total average reward
        best_saver -- tf saver object
        sess -- tf session
        """
        self.best_model_reward = test_reward
        best_model_reward_file = self.folder / 'model_best/best_model_reward'
        with best_model_reward_file.open('w') as f:
            f.write(str(self.best_model_reward))

        best_file = self.folder / 'model_best'
        best_file /= '{}_checkpoint'.format(self.name.replace('-', '_'))
        best_saver.save(sess, str(best_file))

    def choose_action_with_high_confidence(self, pi_values, exclude_noop=True):
        max_confidence_action = np.argmax(pi_values[1 if exclude_noop else 0:])
        confidence = pi_values[max_confidence_action]
        return (max_confidence_action+(1 if exclude_noop else 0)), confidence

    def test_game_classifier(self, global_t, max_steps, sess, worker=None):
        """Evaluate game with current network model.

        Keyword argument:
        sess -- tf session
        """
        logger.info("Testing classifier at global_t={}...".format(global_t))
        self.game_state.reset(hard_reset=True)

        # max_steps = 10000
        total_reward = 0
        total_steps = 0
        episode_reward = 0
        episode_steps = 0
        n_episodes = 0
        while max_steps > 0:
            state = cv2.resize(self.game_state.s_t,
                               self.net.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)
            model_pi = self.net.run_policy(sess, state)
            action, confidence = self.choose_action_with_high_confidence(
                model_pi, exclude_noop=False)

            # take action
            self.game_state.step(action)
            terminal = self.game_state.terminal
            episode_reward += self.game_state.reward
            episode_steps += 1
            max_steps -= 1

            # s_t = s_t1
            self.game_state.update()

            if terminal:
                was_real_done = get_wrapper_by_name(
                    self.game_state.env, 'EpisodicLifeEnv').was_real_done

                if was_real_done:
                    n_episodes += 1
                    score_str = colored("score={}".format(
                        episode_reward), "magenta")
                    steps_str = colored("steps={}".format(
                        episode_steps), "blue")
                    log_data = (n_episodes, score_str, steps_str, self.thread_idx,
                                total_steps)
                    logger.debug("classifier test: trial={} {} {} worker={} "
                                 "total_steps={}"
                                 .format(*log_data))
                    total_reward += episode_reward
                    total_steps += episode_steps
                    episode_reward = 0
                    episode_steps = 0

                self.game_state.reset(hard_reset=False)

        if n_episodes == 0:
            total_reward = episode_reward
            total_steps = episode_steps
        else:
            total_reward = total_reward / n_episodes
            total_steps = total_steps // n_episodes

        log_data = (total_reward, total_steps, n_episodes)
        logger.info("classifier test: final score={} final steps={} # trials={}"
                    .format(*log_data))
        self.record_summary(
            score=total_reward, steps=total_steps,
            episodes=n_episodes, global_t=global_t, mode='Classifier_Test')
        # return log_data

    def train(self, sess, exp_buffer, classify_ctr):
        """Train classification with human demonstration."""
        self.max_val = -(sys.maxsize)
        prev_loss = 0

        for i in range(self.train_max_steps + 1):
            if self.stop_requested:
                break

            #TODO: oversample
            sample = exp_buffer.sample(self.batch_size, beta=0.4)
            index_list, batch, weights = sample
            batch_si, batch_a, batch_returns, batch_fullstate = batch

            feed_dict = {self.net.s: batch_si, self.net.a: batch_a}

            if self.net.use_slv:
                feed_dict[self.net.returns] = batch_returns

            train_loss, acc, max_value, _ = sess.run([
                self.net.total_loss,
                self.net.accuracy,
                self.net.max_value,
                self.apply_gradients],
                feed_dict=feed_dict)

            if max_value > self.max_val:
                self.max_val = max_value

            percent_change = percent_decrease(train_loss, prev_loss)
            # if i % self.eval_freq == 0 or percent_change > 2.5:
            if i % 1000 == 0 or percent_change > 2.5:
                prev_loss = train_loss
                test_acc = sess.run(
                    self.net.accuracy,
                    feed_dict={
                        self.net.s: self.test_batch_si,
                        self.net.a: self.test_batch_a})

                # summary = self.tf.Summary()
                # summary.value.add(tag='Train_Loss',
                #                   simple_value=float(train_loss))
                # summary.value.add(tag='Accuracy', simple_value=float(acc))

                logger.debug("classification: i={0:} train_acc={1:.4f} test_acc={2:.4f}"
                             " loss={3:.4f} max_val={4:}".format(
                              i, acc, test_acc, train_loss, self.max_val))

                # summary_writer.add_summary(summary, i)
                # summary_writer.flush()

        # # Test final network on how well it plays the actual game
        # total_reward, total_steps, n_episodes = self.test_game(sess)
        # summary.value.add(tag='Reward', simple_value=total_reward)
        # summary.value.add(tag='Steps', simple_value=total_steps)
        # summary.value.add(tag='Episodes', simple_value=n_episodes)

        classify_ctr += 1
        log_data = (classify_ctr, len(exp_buffer))
        logger.info("Classifier: #updates={} "
                    "exp_buffer_size={}".format(*log_data))

        # if total_reward >= self.best_model_reward:
        #     self.save_best_model(total_reward, best_saver, sess)

        return classify_ctr

    def testing(self, sess, max_steps, global_t, folder,
        demo_memory_cam=None, worker=None):
        """Evaluate A3C."""
        if self.classify_thread:
            assert worker is not None

        logger.info("Evaluate policy at global_t={}...".format(global_t))

        # copy weights from shared to local
        sess.run(worker.sync)

        if demo_memory_cam is not None and global_t % 5000000 == 0:
            worker.generate_cam_video(sess, 0.03, global_t, folder,
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

        if worker.use_sil and not worker.sil_thread:
            # ensure no states left from a non-terminating episode
            worker.episode.reset()
        return (total_reward, total_steps, n_episodes)
