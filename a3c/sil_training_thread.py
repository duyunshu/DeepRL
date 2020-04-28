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
from queue import Queue, PriorityQueue
from copy import deepcopy
from common_worker import CommonWorker
from sil_memory import SILReplayMemory
from datetime import datetime

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
                 device=None, batch_size=None,
                 use_rollout=False, train_classifier=False, use_sil_neg=False):
        """Initialize A3CTrainingThread class."""
        assert self.action_size != -1

        self.is_sil_thread = True
        self.thread_idx = thread_index
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_input = learning_rate_input
        self.local_net = local_net
        self.batch_size = batch_size
        self.use_rollout=use_rollout
        self.train_classifier=train_classifier
        self.use_sil_neg = use_sil_neg
        # self._device = device

        logger.info("===SIL thread_index: {}===".format(self.thread_idx))
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
            logger.info("sil w_loss: {}".format(w_loss))
            logger.info("sil critic_lr: {}".format(critic_lr))
            logger.info("sil entropy_beta: {}".format(entropy_beta))

            self.local_net.prepare_sil_loss(entropy_beta=entropy_beta,
                                            w_loss=w_loss,
                                            critic_lr=critic_lr,
                                            min_batch_size=min_batch_size,
                                            use_sil_neg = self.use_sil_neg)

            var_refs = [v._ref() for v in local_vars()]

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

        self.episode = SILReplayMemory(
            self.action_size, max_len=None, gamma=self.gamma,
            clip=reward_clipped,
            height=self.local_net.in_shape[0],
            width=self.local_net.in_shape[1],
            phi_length=self.local_net.in_shape[2],
            reward_constant=self.reward_constant)

        self.pick_samples = SILReplayMemory(
            self.action_size, max_len=self.batch_size*2, gamma=self.gamma,
            clip=reward_clipped,
            height=self.local_net.in_shape[0],
            width=self.local_net.in_shape[1],
            phi_length=self.local_net.in_shape[2],priority=True,
            reward_constant=self.reward_constant)

        with tf.device(device):
            if self.use_grad_cam:
                self.game_state = GameState(env_id=self.env_id, display=False,
                                            no_op_max=30, human_demo=False,
                                            episode_life=True, override_num_noops=None)
                self.action_meaning = self.game_state.env.unwrapped \
                    .get_action_meanings()
                self.local_net.build_grad_cam_grads()


    def record_sil(self, sil_ctr=0, total_used=0, num_a3c_used=0, rollout_sampled=0,
                   rollout_used=0, goods=0, bads=0, global_t=0, mode='SIL'):
        """Record SIL."""
        summary = tf.Summary()
        summary.value.add(tag='{}/sil_ctr'.format(mode),
                          simple_value=float(sil_ctr))

        summary.value.add(tag='{}/total_num_sample_used'.format(mode),
                          simple_value=float(total_used))
        summary.value.add(tag='{}/num_a3c_used'.format(mode),
                          simple_value=float(num_a3c_used))
        summary.value.add(tag='{}/num_rollout_sampled'.format(mode),
                          simple_value=float(rollout_sampled))
        summary.value.add(tag='{}/num_rollout_used'.format(mode),
                          simple_value=float(rollout_used))
        summary.value.add(tag='{}/goodstate_size'.format(mode),
                          simple_value=float(goods))
        summary.value.add(tag='{}/badstate_size'.format(mode),
                          simple_value=float(bads))

        self.writer.add_summary(summary, global_t)
        self.writer.flush()

    def sil_train(self, sess, global_t, sil_memory, sil_ctr, m=4,
                  rollout_buffer=None, rollout_proportion=0, stop_rollout=False,
                  roll_any=False):
        """Self-imitation learning process."""

        # copy weights from shared to local
        sess.run(self.sync)
        cur_learning_rate = self._anneal_learning_rate(
                global_t,
                self.initial_learning_rate)

        badstate_queue = Queue()
        goodstate_queue = Queue()

        total_used = 0
        num_a3c_used = 0
        num_rollout_sampled = 0
        num_rollout_used = 0

        for _ in range(m):
            r_batch_size = 0
            if rollout_buffer is not None and len(rollout_buffer) > self.batch_size:
                r_batch_size = self.batch_size
                r_sample = rollout_buffer.sample(r_batch_size, beta=0.4)
                r_index_list, r_batch, r_weights = r_sample
                r_batch_state, r_action, r_batch_returns, \
                    r_batch_fullstate, r_batch_rollout = r_batch
                r_batch_action = []
                for a in r_action:
                    r_batch_action.append(np.argmax(a))
                self.pick_samples.extend_one_priority(r_batch_state,
                    r_batch_fullstate, r_batch_action, r_batch_returns,
                    r_batch_rollout)

            s_batch_size = self.batch_size
            s_sample = sil_memory.sample(s_batch_size , beta=0.4)
            s_index_list, s_batch, s_weights = s_sample
            s_batch_state, s_action, s_batch_returns, \
                s_batch_fullstate, s_batch_rollout = s_batch
            s_batch_action = []
            for a in s_action:
                s_batch_action.append(np.argmax(a))
            self.pick_samples.extend_one_priority(s_batch_state, s_batch_fullstate,
                s_batch_action, s_batch_returns, s_batch_rollout)

            # pick 32 out of 64
            sample = self.pick_samples.sample(self.batch_size, beta=0.4)
            index_list, batch, weights = sample
            batch_state, batch_action, batch_returns, \
                batch_fullstate, batch_rollout = batch

            if self.use_rollout:
                num_rollout_sampled += np.sum(batch_rollout)

            feed_dict = {
                self.local_net.s: batch_state,
                self.local_net.a_sil: batch_action,
                self.local_net.returns: batch_returns,
                self.local_net.weights: weights,
                self.learning_rate_input: cur_learning_rate,
                }
            fetch = [
                self.local_net.clipped_advs,
                self.local_net.advs,
                self.sil_apply_gradients,
                self.sil_apply_gradients_local,
                ]
            adv_clip, adv, _, _ = sess.run(fetch, feed_dict=feed_dict)
            # logger.info("adv_clip: {}".format(adv_clip))
            # logger.info("adv_raw: {}".format(adv))
            pos_idx = [i for (i, num) in enumerate(adv) if num > 0]
            neg_idx = [i for (i, num) in enumerate(adv) if num <= 0]
            total_used += len(pos_idx)
            num_rollout_used += np.sum(np.take(batch_rollout, pos_idx))
            num_a3c_used += (len(pos_idx) - np.sum(np.take(batch_rollout, pos_idx)))
            if self.use_sil_neg:
                total_used += len(neg_idx)

            # update priority
            self.update_priorities_once(sess, sil_memory, s_index_list,
                                        s_batch_state, s_action, s_batch_returns)
            if r_batch_size > 0:
                self.update_priorities_once(sess, rollout_buffer, r_index_list,
                                            r_batch_state, r_action, r_batch_returns)

            # find the index of good samples
            if self.train_classifier:
                for i in pos_idx:
                    goodstate_queue.put(
                        (deepcopy([ batch_state[i] ]),
                         deepcopy([ batch_fullstate[i] ]),
                         deepcopy([ np.argmax(batch_action[i]) ]),
                         deepcopy([ batch_returns[i] ]),
                         deepcopy([ batch_rollout[i] ]))
                        )
            # find the index of bad sampless
            # if we are rolling out specifically from bad states only
            # otherwise, just let rollout sample from shared_memory
            if self.use_rollout and not stop_rollout and not roll_any:
                for i in neg_idx:
                    badstate_queue.put(
                         (  adv[i], #first sort by adv
                            -(int(datetime.now().second + datetime.now().microsecond)),  #if adv same, sort by newest first
                            (deepcopy(batch_state[i]),
                             deepcopy(batch_fullstate[i]),
                             deepcopy(np.argmax(batch_action[i])),
                             deepcopy(batch_returns[i]),
                             deepcopy(batch_rollout[i]))
                         )
                        )

            sil_ctr += 1

        return sil_ctr, total_used, num_a3c_used, num_rollout_sampled, num_rollout_used, \
                goodstate_queue, badstate_queue

    def update_priorities_once(self, sess, memory, index_list, batch_state,
        batch_action, batch_returns):
        """Self-imitation update priorities once."""
        # copy weights from shared to local
        sess.run(self.sync)

        feed_dict = {
            self.local_net.s: batch_state,
            self.local_net.a_sil: batch_action,
            self.local_net.returns: batch_returns,
            }
        fetch = self.local_net.clipped_advs
        adv_clip = sess.run(fetch, feed_dict=feed_dict)
        memory.set_weights(index_list, adv_clip)


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
