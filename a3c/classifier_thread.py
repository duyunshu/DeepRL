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

logger = logging.getLogger("classifier_thread")


class ClassifyDemo(CommonWorker):
    """Use Supervised learning for learning features."""

    def __init__(self, tf, net, thread_index, game_name, train_max_steps, batch_size,
                 ae_grad_applier=None, grad_applier=None, eval_freq=5000,
                 demo_memory_folder=None, demo_ids=None, folder=None,
                 exclude_num_demo_ep=0, clip_norm=None, game_state=None,
                 sampling_type=None, sl_loss_weight=1.0,
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

        logger.info("===CLASSIFIER thread_index {}===".format(self.thread_idx))
        logger.info("train_max_steps: {}".format(self.train_max_steps))
        logger.info("batch_size: {}".format(self.batch_size))
        logger.info("eval_freq: {}".format(self.eval_freq))
        logger.info("sampling_type: {}".format(self.sampling_type))
        logger.info("clip_norm: {}".format(clip_norm))

        self.net.prepare_loss(sl_loss_weight=1.0, val_weight=0.5)
        self.net.prepare_evaluate()
        self.apply_gradients = self.prepare_compute_gradients(
            grad_applier, clip_norm=clip_norm)

        sample = exp_buffer.sample(512, beta=0.4)
        self.test_index_list, batch, _ = sample
        batch_state, batch_action, _, _, _ = batch

        self.test_batch_si = deepcopy(batch_state)
        self.test_batch_a = deepcopy(batch_action)

        self.demo_size = len(exp_buffer) - 512

        del batch_state, batch_action

    def prepare_compute_gradients(self, grad_applier, clip_norm=None):
        """Return operation for gradient application.

        Keyword arguments:
        grad_applier -- optimizer for applying gradients
        device -- cpu or gpu
        clip_norm -- value for clip_by_global_norm (default None)
        """
        with self.net.graph.as_default():
            # with self.tf.device(device):
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

    def train(self, sess, global_t, exp_buffer, classify_ctr):
        """Train classification with human demonstration."""
        self.max_val = -(sys.maxsize)
        prev_loss = 0

        for i in range(self.train_max_steps + 1):
            if self.stop_requested:
                break

            #TODO: oversample
            sample = exp_buffer.sample(self.batch_size, beta=0.4)
            index_list, batch, weights = sample
            batch_si, batch_a, batch_returns, batch_fullstate, _ = batch

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

                logger.debug("classification: i={0:} train_acc={1:.4f} test_acc={2:.4f}"
                             " loss={3:.4f} max_val={4:}".format(
                              i, acc, test_acc, train_loss, self.max_val))

        classify_ctr += 1
        log_data = (classify_ctr, len(exp_buffer))
        logger.info("Classifier: #updates={} "
                    "exp_buffer_size={}".format(*log_data))

        # if total_reward >= self.best_model_reward:
        #     self.save_best_model(total_reward, best_saver, sess)

        return classify_ctr
