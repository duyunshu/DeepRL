#!/usr/bin/env python3
import cv2
import logging
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

from common.game_state import GameState
from common.game_state import get_wrapper_by_name
from common.util import transform_h
from common.util import transform_h_inv
from termcolor import colored
from queue import Queue
from copy import deepcopy
from common_worker import CommonWorker

logger = logging.getLogger("sil_training_thread")


class SILTrainingThread(CommonWorker):
    """Asynchronous Actor-Critic Training Thread Class."""

    entropy_beta = 0.01
    gamma = 0.99
    finetune_upper_layers_only = False
    transformed_bellman = False
    clip_norm = 0.5
    use_grad_cam = False
    use_sil_neg = False # test if also using samples that are (G-V<0)

    def __init__(self, thread_index, global_net, local_net,
                 initial_learning_rate, learning_rate_input, grad_applier,
                 max_global_time_step, device=None,
                 batch_size=None, no_op_max=30):
        """Initialize A3CTrainingThread class."""
        assert self.action_size != -1

        self.is_sil_thread = True
        self.thread_idx = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step
        self.local_net = local_net
        self.batch_size = batch_size

        self.no_op_max = no_op_max
        self.override_num_noops = 0 if self.no_op_max == 0 else None

        logger.info("===SIL thread_index: {}".format(self.thread_idx))
        logger.info("device: {}".format(device))

        logger.info("use_sil_neg: {}".format(
            colored(self.use_sil_neg, "green" if self.use_sil_neg else "red")))

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
        logger.info("use_grad_cam: {}".format(colored(self.use_grad_cam,
                    "green" if self.use_grad_cam else "red")))

        reward_clipped = True if self.reward_type == 'CLIP' else False
        local_vars = self.local_net.get_vars
        if self.finetune_upper_layers_only:
            local_vars = self.local_net.get_vars_upper

        with tf.device(device):
            # TODO(add as command-line parameters later)
            critic_lr = 0.1
            entropy_beta = 0
            w_loss = 1.0

            min_batch_size = 1
            # if reward_clipped:
            #     min_batch_size = int(np.ceil(self.batch_size / 16) * 2)
            #     critic_lr = 0.01

            logger.info("sil batch_size: {}".format(self.batch_size))
            logger.info("sil min_batch_size: {}".format(min_batch_size))
            logger.info("sil w_loss: {}".format(critic_lr))
            logger.info("sil critic_lr: {}".format(critic_lr))
            logger.info("sil entropy_beta: {}".format(entropy_beta))
            self.local_net.prepare_sil_loss(entropy_beta=entropy_beta,
                                            w_loss=w_loss,
                                            critic_lr=critic_lr,
                                            min_batch_size=min_batch_size,
                                            use_sil_neg = self.use_sil_neg)
            self.sil_gradients = tf.gradients(
                self.local_net.total_loss_sil, var_refs)

        global_vars = global_net.get_vars
        if self.finetune_upper_layers_only:
            global_vars = global_net.get_vars_upper

        with tf.device(device):
            if self.clip_norm is not None:
                self.sil_gradients, grad_norm = tf.clip_by_global_norm(
                    self.sil_gradients, self.clip_norm)
            sil_gradients_global = list(
                zip(self.sil_gradients, global_vars()))
            sil_gradients_local = list(
                zip(self.sil_gradients, local_vars()))
            self.sil_apply_gradients = grad_applier.apply_gradients(
                sil_gradients_global)
            self.sil_apply_gradients_local = grad_applier.apply_gradients(
                sil_gradients_local)

        self.sync = self.local_net.sync_from(
            global_net, upper_layers_only=self.finetune_upper_layers_only)

        self.game_state = GameState(env_id=self.env_id, display=False,
                                    no_op_max=self.no_op_max, human_demo=False,
                                    episode_life=True,
                                    override_num_noops=self.override_num_noops)

        self.local_t = 0

        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0
        self.episode_steps = 0

        # variable controlling log output
        self.prev_local_t = 0

        with tf.device(device):
            if self.use_grad_cam:
                self.action_meaning = self.game_state.env.unwrapped \
                    .get_action_meanings()
                self.local_net.build_grad_cam_grads()

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate \
            * (self.max_global_time_step - global_time_step) \
            / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def sil_train(self, sess, global_t, sil_memory, sil_ctr, m=4):
        """Self-imitation learning process."""

        # copy weights from shared to local
        sess.run(self.sync)
        cur_learning_rate = self._anneal_learning_rate(global_t)

        goodstate_queue = Queue()
        badstate_queue = Queue()

        for _ in range(m):
            sample = sil_memory.sample(self.batch_size, beta=0.4)
            index_list, batch, weights = sample
            batch_state, batch_action, batch_returns, batch_fullstate = batch

            feed_dict = {
                self.local_net.s: batch_state,
                self.local_net.a_sil: batch_action,
                self.local_net.returns: batch_returns,
                self.local_net.weights: weights,
                self.learning_rate_input: cur_learning_rate,
                }
            fetch = [
                self.local_net.clipped_advs,
                self.sil_apply_gradients,
                self.sil_apply_gradients_local,
                ]
            adv, _, _ = sess.run(fetch, feed_dict=feed_dict)
            sil_memory.set_weights(index_list, adv) # only exec if priority=True
            # logger.info("advantage: {}".format(adv))

            # find the index of samples
            if self.use_correction:
                for i in range(len(index_list)):
                    if adv[i] <= 0.0:
                        badstate_queue.put(
                            (deepcopy(batch_state[i]),
                             deepcopy(batch_fullstate[i]),
                             deepcopy(np.argmax(batch_action[i])),
                             deepcopy(batch_returns[i])))
                    else:
                        goodstate_queue.put(
                            (deepcopy([batch_state[i]]),
                             deepcopy([batch_fullstate[i]]),
                             deepcopy([np.argmax(batch_action[i])]),
                             deepcopy([batch_returns[i]])))

            sil_ctr += 1
        return sil_ctr, goodstate_queue, badstate_queue


    def sil_update_priorities(self, sess, sil_memory, m=4):
        """Self-imitation update priorities."""

        # copy weights from shared to local
        sess.run(self.sync)

        # start_time = time.time()
        for _ in range(m):
            sample = sil_memory.sample(self.batch_size, beta=0.4)
            index_list, batch, _ = sample
            batch_state, batch_action, batch_returns, batch_fullstate = batch

            feed_dict = {
                self.local_net.s: batch_state,
                self.local_net.a_sil: batch_action,
                self.local_net.returns: batch_returns,
                }
            fetch = self.local_net.clipped_advs
            adv = sess.run(fetch, feed_dict=feed_dict)
            sil_memory.set_weights(index_list, adv)

