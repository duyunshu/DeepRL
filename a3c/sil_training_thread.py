#!/usr/bin/env python3
import cv2
import logging
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

from common.game_state import GameState
from common.game_state import get_wrapper_by_name
from common.util import convert_onehot_to_a
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
                 use_rollout=False, rollout_sample_proportion=None,
                 train_classifier=False, use_sil_neg=False,
                 one_buffer=False):
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
        self.one_buffer = one_buffer
        self.rollout_sample_proportion = rollout_sample_proportion

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

        # if not self.use_rollout or rollout_sample_proportion is not None:
        #     max_len = self.batch_size
        # else:
        #     max_len = self.batch_size*2
        # init temp buffer, only use temp_buffer in Lider adaptive sampling
        self.temp_buffer = None
        if (self.use_rollout) and (not self.one_buffer) and (self.rollout_sample_proportion is None):
            self.temp_buffer = SILReplayMemory(
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


    def record_sil(self, sil_ctr=0, total_used=0, num_a3c_used=0, a3c_used_return=0,
                   rollout_sampled=0, rollout_used=0, rollout_used_return=0,
                   old_sampled=0, old_used=0, goods=0, bads=0,
                   global_t=0, mode='SIL'):
        """Record SIL."""
        summary = tf.Summary()
        summary.value.add(tag='{}/sil_ctr'.format(mode),
                          simple_value=float(sil_ctr))

        summary.value.add(tag='{}/total_num_sample_used'.format(mode),
                          simple_value=float(total_used))
        summary.value.add(tag='{}/num_a3c_used'.format(mode),
                          simple_value=float(num_a3c_used))
        summary.value.add(tag='{}/a3c_used_return'.format(mode),
                          simple_value=float(a3c_used_return))
        summary.value.add(tag='{}/num_rollout_sampled'.format(mode),
                          simple_value=float(rollout_sampled))
        summary.value.add(tag='{}/num_rollout_used'.format(mode),
                          simple_value=float(rollout_used_return))
        summary.value.add(tag='{}/rollout_used_return'.format(mode),
                          simple_value=float(rollout_used))
        summary.value.add(tag='{}/num_old_sampled'.format(mode),
                          simple_value=float(old_sampled))
        summary.value.add(tag='{}/num_old_used'.format(mode),
                          simple_value=float(old_used))
        summary.value.add(tag='{}/goodstate_size'.format(mode),
                          simple_value=float(goods))
        summary.value.add(tag='{}/badstate_size'.format(mode),
                          simple_value=float(bads))

        self.writer.add_summary(summary, global_t)
        self.writer.flush()

    def _extend_batches(batches, extend_batches):
        batch_state, batch_action, batch_returns, \
            batch_fullstate, batch_rollout, batch_refresh = batches

        state, action, returns, \
            fullstate, rollout, refresh = extend_batches

        batch_state.extend(state)
        batch_action.extend(action)
        batch_returns.extend(returns)
        batch_fullstate.extend(fullstate)
        batch_rollout.extend(rollout)
        batch_refresh.extend(refresh)

        return (batch_state, batch_action, batch_returns,
                batch_fullstate, batch_rollout, batch_refresh)

    def sil_train(self, sess, global_t, sil_memory, m,
                  rollout_buffer=None, stop_rollout=False, roll_any=False):
        """Self-imitation learning process."""

        # copy weights from shared to local
        sess.run(self.sync)
        cur_learning_rate = self._anneal_learning_rate(global_t,
                                                       self.initial_learning_rate)
        badstate_queue = Queue()
        goodstate_queue = Queue()

        local_sil_ctr, local_sil_a3c_sampled, local_sil_a3c_used = 0, 0, 0
        local_sil_a3c_sampled_return, local_sil_a3c_used_return = 0, 0
        local_sil_a3c_used_adv, local_sil_rollout_sampled, local_sil_rollout_used = 0, 0, 0
        local_sil_rollout_sampled_return, local_sil_rollout_used_return = 0, 0
        local_sil_rollout_used_adv, local_sil_old_sampled, local_sil_old_used = 0, 0, 0

        total_used = 0
        num_a3c_used = 0
        num_rollout_sampled = 0
        num_rollout_used = 0

        num_old_sampled = 0
        num_old_used = 0

        for _ in range(m):
            s_batch_size, r_batch_size = 0, 0
            # a3ctbsil
            if not self.use_rollout:
                s_batch_size = self.batch_size
            # or LiDER-OneBuffer
            elif self.use_rollout and self.one_buffer:
                s_batch_size = self.batch_size
            # LiDER
            else:
                assert rollout_buffer is not None
                # adaptive sampling ratios
                if self.rollout_sample_proportion is None:
                    assert self.temp_buffer is not None
                    self.temp_buffer.reset()
                    s_batch_size = self.batch_size
                    r_batch_size = self.batch_size
                # fixed sampling ratio
                else:
                    assert self.rollout_sample_proportion > 0 # rollout proportion=0 is actually just a3tbsil
                    s_batch_size = round(self.batch_size * (1 - self.rollout_sample_proportion))
                    r_batch_size = round(self.batch_size * self.rollout_sample_proportion)

            batch_state, batch_action, batch_returns, batch_fullstate, \
                batch_rollout, batch_refresh, weights = ([] for i in range(7))

            # take from buffer a3c
            if s_batch_size > 0 and len(sil_memory) > s_batch_size:
                s_sample = sil_memory.sample(s_batch_size , beta=0.4)
                s_index_list, s_batch, s_weights = s_sample
                s_batch_state, s_action, s_batch_returns, \
                    s_batch_fullstate, s_batch_rollout, s_batch_refresh = s_batch
                local_sil_a3c_sampled_return += np.sum(s_batch_returns)
                # update priority of sampled experiences
                self.update_priorities_once(sess, sil_memory, s_index_list,
                                            s_batch_state, s_action, s_batch_returns)
                if self.temp_buffer is not None:
                    s_batch_action = convert_onehot_to_a(s_action)
                    self.temp_buffer.extend_one_priority(s_batch_state, s_batch_fullstate,
                        s_batch_action, s_batch_returns, s_batch_rollout, s_batch_refresh)
                else:
                    batch_state.extend(s_batch_state)
                    batch_action.extend(s_action)
                    batch_returns.extend(s_batch_returns)
                    batch_fullstate.extend(s_batch_fullstate)
                    batch_rollout.extend(s_batch_rollout)
                    batch_refresh.extend(s_batch_refresh)
                    weights.extend(s_weights)

            # take from buffer rollout
            if r_batch_size > 0 and len(rollout_buffer) > r_batch_size:
                r_sample = rollout_buffer.sample(r_batch_size, beta=0.4)
                r_index_list, r_batch, r_weights = r_sample
                r_batch_state, r_action, r_batch_returns, \
                    r_batch_fullstate, r_batch_rollout, r_batch_refresh = r_batch
                local_sil_rollout_sampled_return += np.sum(r_batch_returns)
                # update priority of sampled experiences
                self.update_priorities_once(sess, rollout_buffer, r_index_list,
                                            r_batch_state, r_action, r_batch_returns)
                if self.temp_buffer is not None:
                    r_batch_action = convert_onehot_to_a(r_action)
                    self.temp_buffer.extend_one_priority(r_batch_state, r_batch_fullstate,
                        r_batch_action, r_batch_returns, r_batch_rollout, r_batch_refresh)
                else:
                    batch_state.extend(r_batch_state)
                    batch_action.extend(r_action)
                    batch_returns.extend(r_batch_returns)
                    batch_fullstate.extend(r_batch_fullstate)
                    batch_rollout.extend(r_batch_rollout)
                    batch_refresh.extend(r_batch_refresh)
                    weights.extend(r_weights)

            # LiDER only: pick 32 out of mixed
            # (at the beginning 32 could all be a3c since rollout has no data yet)
            # make sure the temp buffer has been filled
            if self.temp_buffer is not None and len(self.temp_buffer) >= self.batch_size:
                sample = self.temp_buffer.sample(self.batch_size, beta=0.4)
                index_list, batch, weights = sample
                batch_state, batch_action, batch_returns, \
                    batch_fullstate, batch_rollout, batch_refresh = batch

            if self.use_rollout:
                num_rollout_sampled += np.sum(batch_rollout)
                num_old_sampled += np.sum(batch_refresh)

            # sil policy update (if one full batch sampled)
            # this control is mostly for rollout because rollout buffer takes time to fill)
            if len(batch_state) == self.batch_size:
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
                num_old_used += np.sum(np.take(batch_refresh, pos_idx))
                if self.use_sil_neg:
                    total_used += len(neg_idx)

                rollout_idx = [i for (i, num) in enumerate(batch_rollout) if num > 0]
                pos_rollout_idx = np.intersect1d(rollout_idx, pos_idx)
                if len(pos_rollout_idx) > 0 and len(pos_rollout_idx) == num_rollout_used:
                    local_sil_rollout_used_return += np.sum(np.take(batch_returns, pos_rollout_idx))
                    local_sil_rollout_used_adv += np.sum(np.take(adv, pos_rollout_idx))

                a3c_idx = [i for (i, num) in enumerate(batch_rollout) if num <= 0]
                pos_a3c_idx = np.intersect1d(a3c_idx, pos_idx)
                if len(pos_a3c_idx) > 0 and len(pos_a3c_idx)==num_a3c_used:
                    local_sil_a3c_used_return += np.sum(np.take(batch_returns, pos_a3c_idx))
                    local_sil_a3c_used_adv += np.sum(np.take(adv, pos_a3c_idx))

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
                # find the index of bad samples if we are rolling out specifically from bad states
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

                local_sil_ctr += 1

        local_sil_a3c_sampled += (self.batch_size*m - num_rollout_sampled)
        local_sil_a3c_used += num_a3c_used
        local_sil_rollout_sampled += num_rollout_sampled
        local_sil_rollout_used += num_rollout_used
        local_sil_old_sampled += num_old_sampled
        local_sil_old_used += num_old_used

        return local_sil_ctr, local_sil_a3c_sampled, local_sil_a3c_used, \
               local_sil_a3c_sampled_return, local_sil_a3c_used_return, local_sil_a3c_used_adv, \
               local_sil_rollout_sampled, local_sil_rollout_used, \
               local_sil_rollout_sampled_return, local_sil_rollout_used_return, local_sil_rollout_used_adv, \
               local_sil_old_sampled, local_sil_old_used, goodstate_queue, badstate_queue

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
