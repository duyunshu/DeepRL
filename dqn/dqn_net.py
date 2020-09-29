#!/usr/bin/env python3
import tensorflow as tf
import random
import numpy as np
import logging
import os
import time

from termcolor import colored
from time import sleep
from net import Network
from common.util import plot_conv_weights, graves_rmsprop_optimizer

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger("dqn_net")

class DqnNet(Network):
    """ DQN Network Model of DQN Algorithm """

    def __init__(
        self, sess, height, width, phi_length, n_actions, name, gamma=0.99,
        optimizer='RMS', learning_rate=0.00025, epsilon=0.01, decay=0.95,
        momentum=0., l2_decay=0.0001, slow=False, tau=0.01, verbose=False,
        folder='_networks',
        transfer=False, transfer_folder='',
        not_transfer_conv2=False, not_transfer_conv3=False,
        not_transfer_fc1=False, not_transfer_fc2=False, device="/cpu:0",
        transformed_bellman=False, target_consistency_loss=False,
        clip_norm=None, weight_decay=None,use_rollout=False, use_sil=False,
        double_q=False):
        """ Initialize network """
        Network.__init__(self, sess, name=name)
        self.gamma = gamma
        self.slow = slow
        self.tau = tau
        self.name = name
        self.sess = sess
        self.folder = folder
        self._device = device
        self.transformed_bellman = transformed_bellman
        self.target_consistency_loss = target_consistency_loss
        self.verbose = verbose
        self.use_rollout = use_rollout
        self.use_sil = use_sil
        self.double_q = double_q

        logger.info("===DQN net===")
        logger.info("gamma: {}".format(self.gamma))
        logger.info("optimizer: {}".format(optimizer))
        logger.info("learning rate: {}".format(learning_rate))
        logger.info("transformed_bellman: {}".format(
            colored(self.transformed_bellman,
                    "green" if self.transformed_bellman else "red")))
        logger.info("double DQN: {}".format(
            colored(self.double_q,
                    "green" if self.double_q else "red")))
        time.sleep(2)

        self.observation = tf.placeholder(tf.float32, [None, height, width, phi_length], name='observation')
        self.observation_n = tf.div(self.observation, 255.)

        self.next_observation = tf.placeholder(tf.float32, [None, height, width, phi_length], name='t_next_observation')
        self.next_observation_n = tf.div(self.next_observation, 255.)

        def q_func_builder(input_placeholder, scope, reuse=False):
            with tf.device(self._device), tf.variable_scope(scope, reuse=reuse) as scope:
                # q network model:
                self.W_conv1, self.b_conv1 = self.conv_variable([8, 8, phi_length, 32], layer_name='conv1', gain=np.sqrt(2))
                self.h_conv1 = tf.nn.relu(tf.add(self.conv2d(input_placeholder, #self.observation_n,
                    self.W_conv1, 4), self.b_conv1), name=self.name + '_conv1_activations')
                tf.add_to_collection('conv_weights', self.W_conv1)
                tf.add_to_collection('conv_output', self.h_conv1)

                self.W_conv2, self.b_conv2 = self.conv_variable([4, 4, 32, 64], layer_name='conv2', gain=np.sqrt(2))
                self.h_conv2 = tf.nn.relu(tf.add(self.conv2d(self.h_conv1, self.W_conv2, 2), self.b_conv2), name=self.name + '_conv2_activations')
                tf.add_to_collection('conv_weights', self.W_conv2)
                tf.add_to_collection('conv_output', self.h_conv2)

                self.W_conv3, self.b_conv3 = self.conv_variable([3, 3, 64, 64], layer_name='conv3', gain=np.sqrt(2))
                self.h_conv3 = tf.nn.relu(tf.add(self.conv2d(self.h_conv2, self.W_conv3, 1), self.b_conv3), name=self.name + '_conv3_activations')
                tf.add_to_collection('conv_weights', self.W_conv3)
                tf.add_to_collection('conv_output', self.h_conv3)

                self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 3136])

                self.W_fc1, self.b_fc1 = self.fc_variable([3136, 512], layer_name='fc1', gain=np.sqrt(2))
                self.h_fc1 = tf.nn.relu(tf.add(tf.matmul(self.h_conv3_flat, self.W_fc1), self.b_fc1), name=self.name + '_fc1_activations')

                self.W_fc2, self.b_fc2 = self.fc_variable([512, n_actions], layer_name='fc2')
                #self.
                q_value = tf.add(tf.matmul(self.h_fc1, self.W_fc2), self.b_fc2, name=self.name + '_fc1_outputs')
                return q_value

        self.q_value = q_func_builder(self.observation_n, scope='net_-1')
        if self.double_q:
            self.double_q_value = q_func_builder(self.next_observation_n, scope='net_-1', reuse=True)

        if self.target_consistency_loss:
            self.tc_observation = tf.placeholder(tf.float32, [None, height, width, phi_length], name='observation_tc')
            self.tc_observation_n = tf.div(self.tc_observation, 255.)

            with tf.device(self._device), tf.variable_scope('net_-1', reuse=True) as scope:
                # q network model:
                tc_W_conv1, tc_b_conv1 = self.conv_variable([8, 8, phi_length, 32], layer_name='conv1', gain=np.sqrt(2))
                tc_h_conv1 = tf.nn.relu(tf.add(self.conv2d(self.tc_observation_n, tc_W_conv1, 4), tc_b_conv1), name=self.name + '_conv1_activations')

                tc_W_conv2, tc_b_conv2 = self.conv_variable([4, 4, 32, 64], layer_name='conv2', gain=np.sqrt(2))
                tc_h_conv2 = tf.nn.relu(tf.add(self.conv2d(tc_h_conv1, tc_W_conv2, 2), tc_b_conv2), name=self.name + '_conv2_activations')

                tc_W_conv3, tc_b_conv3 = self.conv_variable([3, 3, 64, 64], layer_name='conv3', gain=np.sqrt(2))
                tc_h_conv3 = tf.nn.relu(tf.add(self.conv2d(tc_h_conv2, tc_W_conv3, 1), tc_b_conv3), name=self.name + '_conv3_activations')

                tc_h_conv3_flat = tf.reshape(tc_h_conv3, [-1, 3136])

                tc_W_fc1, tc_b_fc1 = self.fc_variable([3136, 512], layer_name='fc1', gain=np.sqrt(2))
                tc_h_fc1 = tf.nn.relu(tf.add(tf.matmul(tc_h_conv3_flat, tc_W_fc1), tc_b_fc1), name=self.name + '_fc1_activations')

                tc_W_fc2, tc_b_fc2 = self.fc_variable([512, n_actions], layer_name='fc2')
                self.tc_q_value = tf.add(tf.matmul(tc_h_fc1, tc_W_fc2), tc_b_fc2, name=self.name + '_fc1_outputs')

        if transfer:
            self.load_transfer_model(
                self.sess, folder=transfer_folder,
                not_transfer_fc2=not_transfer_fc2, not_transfer_fc1=not_transfer_fc1,
                not_transfer_conv3=not_transfer_conv3, not_transfer_conv2=not_transfer_conv2)

        if self.verbose:
            self.init_verbosity()


        with tf.device(self._device), tf.variable_scope('net_-1-target') as scope:
            # target q network model:
            kernel_shape = [8, 8, phi_length, 32]
            self.t_W_conv1, self.t_b_conv1 = self.conv_variable(kernel_shape, layer_name='t_conv1')
            self.t_h_conv1 = tf.nn.relu(tf.add(self.conv2d(self.next_observation_n, self.t_W_conv1, 4), self.t_b_conv1), name=self.name + '_t_conv1_activations')

            kernel_shape = [4, 4, 32, 64]
            self.t_W_conv2, self.t_b_conv2 = self.conv_variable(kernel_shape, layer_name='t_conv2')
            self.t_h_conv2 = tf.nn.relu(tf.add(self.conv2d(self.t_h_conv1, self.t_W_conv2, 2), self.t_b_conv2), name=self.name + '_t_conv2_activations')

            kernel_shape = [3, 3, 64, 64]
            self.t_W_conv3, self.t_b_conv3 = self.conv_variable(kernel_shape, layer_name='t_conv3')
            self.t_h_conv3 = tf.nn.relu(tf.add(self.conv2d(self.t_h_conv2, self.t_W_conv3, 1), self.t_b_conv3), name=self.name + '_t_conv3_activations')

            self.t_h_conv3_flat = tf.reshape(self.t_h_conv3, [-1, 3136])

            kernel_shape = [3136, 512]
            self.t_W_fc1, self.t_b_fc1 = self.fc_variable(kernel_shape, layer_name='t_fc1')
            self.t_h_fc1 = tf.nn.relu(tf.add(tf.matmul(self.t_h_conv3_flat, self.t_W_fc1), self.t_b_fc1), name=self.name + '_t_fc1_activations')

            kernel_shape = [512, n_actions]
            self.t_W_fc2, self.t_b_fc2 = self.fc_variable(kernel_shape, layer_name='t_fc2')
            self.t_q_value = tf.add(tf.matmul(self.t_h_fc1, self.t_W_fc2), self.t_b_fc2, name=self.name + '_t_fc1_outputs')

        with tf.device(self._device):
            # cost of q network
            self.cost, self.td_error = self.build_loss(n_actions) #+ self.l2_regularizer_loss
            # sil cost for rollout
            if self.use_rollout or self.use_sil:
                self.td_loss, self.masked_td_loss, self.sil_td_error, \
                    self.clipped_sil_td_error, self.sil_cost = self.build_sil_loss(n_actions)

            with tf.name_scope("Train") as scope:
                if optimizer == "Adam":
                    self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
                elif optimizer == "AdamW":
                    assert weight_decay is not None
                    self.opt = tf.contrib.opt.AdamWOptimizer(weight_decay=weight_decay, learning_rate=learning_rate, epsilon=epsilon)
                elif optimizer == "RMS":
                    # Tensorflow RMSOptimizer
                    if weight_decay is None:
                        self.opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=momentum, epsilon=epsilon)
                    else:
                        RMSPropW = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.RMSPropOptimizer)
                        self.opt = RMSPropW(weight_decay=weight_decay, learning_rate=learning_rate, decay=decay, momentum=momentum, epsilon=epsilon)
                else:
                    logger.error("Unknown Optimizer!")
                    sys.exit()

                var_refs = [v._ref() for v in self.get_vars()]
                gradients = tf.gradients(self.cost, var_refs)
                if clip_norm is not None:
                    gradients, grad_norm = tf.clip_by_global_norm(gradients, clip_norm)
                gradients = list(zip(gradients, self.get_vars()))
                self.train_step = self.opt.apply_gradients(gradients)

                # sil cost for rollout
                if self.use_rollout or self.use_sil:
                    sil_var_refs = [v._ref() for v in self.get_vars()]
                    sil_gradients = tf.gradients(self.sil_cost, sil_var_refs)
                    if clip_norm is not None:
                        sil_gradients, sil_grad_norm = tf.clip_by_global_norm(sil_gradients, clip_norm)
                    sil_gradients = list(zip(sil_gradients, self.get_vars()))
                    self.sil_train_step = self.opt.apply_gradients(sil_gradients)

        def initialize_uninitialized(sess):
            global_vars = tf.global_variables()
            is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))

        if transfer:
            initialize_uninitialized(self.sess)
        else:
            # initialize all tensor variable parameters
            self.sess.run(tf.global_variables_initializer())

        # Make sure q and target model have same initial parameters copy the parameters
        self.update_target_network(slow=False)
        logger.info("target model assigned the same parameters as q model")

        self.saver = tf.train.Saver()
        if self.folder is not None:
            self.summary_op = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter('results/log/dqn/{}/'.format(self.name.replace('-', '_')) + self.folder[12:], self.sess.graph)

    def update_target_network(self, name=None, slow=False):
        logger.info('update target network')
        self.sess.run(self.update_target_op(name=name, slow=slow))

    def sync_rollout_network(self, src_network=None, name=None, slow=False):
        # op for rollout net only
        assert self.is_rollout_net == True
        assert src_network is not None
        logger.info('sync rollout network with global network')
        self.sync_from(src_network=src_network, name=name, slow=False)

    def get_vars(self):
        return [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_conv3, self.b_conv3,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2]

    def get_target_vars(self):
        return [
            self.t_W_conv1, self.t_b_conv1,
            self.t_W_conv2, self.t_b_conv2,
            self.t_W_conv3, self.t_b_conv3,
            self.t_W_fc1, self.t_b_fc1,
            self.t_W_fc2, self.t_b_fc2]

    def update_target_op(self, name=None, slow=False):
        src_vars = self.get_vars()
        dst_vars = self.get_target_vars()

        sync_ops = []
        with tf.device(self._device):
            with tf.name_scope(name, "DqnNet", []) as name:
                for (src_var, dst_var) in zip(src_vars, dst_vars):
                    if slow:
                        slow_src_var = tf.multiply(dst_var, 1 - self.tau) + tf.multiply(src_var, self.tau)
                        sync_op = tf.assign(dst_var, slow_src_var)
                    else:
                        sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)

    def sync_from(self, src_network, name=None, slow=False):
        src_vars = src_network.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []
        with tf.device(self._device):
            with tf.name_scope(name, "DqnNet", []) as name:
                for(src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)

    def evaluate(self, state):
        return self.sess.run(self.q_value, feed_dict={self.observation: [state]})

    def evaluate_target(self, state):
        return self.sess.run(self.t_q_value, feed_dict={self.next_observation: [state]})

    def evaluate_tc(self, state):
        return self.sess.run(self.q_value, feed_dict={self.observation: state})

    def build_loss(self, n_actions):
        with tf.name_scope("Loss") as scope:
            self.actions = tf.placeholder(tf.float32, shape=[None, n_actions], name="actions") # one-hot matrix
            self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
            self.terminals = tf.placeholder(tf.float32, shape=[None], name="terminals")
            self.imp_weights = tf.placeholder(tf.float32, [None], name="weight")

            # q scores for actions which we know were selected in the given state.
            predictions = tf.reduce_sum(tf.multiply(self.q_value, self.actions), axis=1)

            if self.double_q:
                # double_q_value = tf.reduce_sum(tf.multiply(self.double_q_value, self.actions), axis=1)
                max_double_q_action_values = tf.argmax(self.double_q_value , axis=1)
                max_action_values = tf.reduce_sum(tf.multiply(self.t_q_value,
                                                              tf.one_hot(max_double_q_action_values,
                                                              n_actions)), axis=1)
            else:
                max_action_values = tf.reduce_max(self.t_q_value, axis=1)

            def h(z, eps=10**-2):
                return (tf.sign(z) * (tf.sqrt(tf.abs(z) + 1.) - 1.)) + (eps * z)

            def h_inv(z, eps=10**-2):
                return tf.sign(z) * (tf.square((tf.sqrt(1 + 4 * eps * (tf.abs(z) + 1 + eps)) - 1) / (2 * eps)) - 1)

            def h_log(z, eps=1):
                return (tf.sign(z) * tf.log(1. + tf.abs(z)) * eps)

            def h_inv_log(z, eps=1):
                return tf.sign(z) * (tf.math.exp(tf.abs(z) / eps) - 1)

            if self.transformed_bellman:
                transformed = h(self.rewards + self.gamma * h_inv(max_action_values) * (1 - self.terminals))
                targets = transformed
            else:
                targets = self.rewards + (self.gamma * max_action_values * (1 - self.terminals))

            td_error = predictions - tf.stop_gradient(targets)
            td_loss = tf.losses.huber_loss(
                tf.stop_gradient(targets),
                predictions,
                reduction=tf.losses.Reduction.NONE)
            # weighted_loss = tf.reduce_mean(self.imp_weights * td_loss)

            if self.target_consistency_loss:
                t_actions_one_hot = tf.one_hot(tf.argmax(self.t_q_value, axis=1), n_actions, 1.0, 0.0)
                max_action_values_q = tf.reduce_sum(tf.multiply(self.tc_q_value, t_actions_one_hot), axis=1)
                tc_loss = tf.losses.huber_loss(
                    tf.stop_gradient(max_action_values),
                    max_action_values_q,
                    reduction=tf.losses.Reduction.NONE)
                total_loss = tf.reduce_mean(self.imp_weights * td_loss + tc_loss)
            else:
                total_loss = tf.reduce_mean(self.imp_weights * td_loss)

            loss = total_loss

            tf.summary.scalar("loss", loss)
            tf.summary.scalar("td_error", tf.reduce_max(td_error))
            tf.summary.scalar("cost_min", tf.reduce_max(td_loss))
            tf.summary.scalar("cost_max", tf.reduce_max(td_loss))
            tf.summary.scalar("target_min", tf.reduce_min(targets))
            tf.summary.scalar("target_max", tf.reduce_max(targets))
            tf.summary.scalar("acted_Q_min", tf.reduce_min(predictions))
            tf.summary.scalar("acted_Q_max", tf.reduce_max(predictions))
            tf.summary.scalar("reward_max", tf.reduce_max(self.rewards))
            return loss, td_error


    def build_sil_loss(self, n_actions, min_batch_size=1):
        with tf.name_scope("SIL_Loss") as scope:
            self.sil_actions = tf.placeholder(tf.float32, shape=[None, n_actions], name="sil_actions") # one-hot matrix
            self.sil_rewards = tf.placeholder(tf.float32, shape=[None], name="sil_rewards")
            self.sil_terminals = tf.placeholder(tf.float32, shape=[None], name="sil_terminals")
            self.sil_imp_weights = tf.placeholder(tf.float32, [None], name="sil_weight")
            predictions = tf.reduce_sum(tf.multiply(self.q_value, self.sil_actions), axis=1)
            max_action_values = tf.reduce_max(self.t_q_value, axis=1)

            def h(z, eps=10**-2):
                return (tf.sign(z) * (tf.sqrt(tf.abs(z) + 1.) - 1.)) + (eps * z)

            def h_inv(z, eps=10**-2):
                return tf.sign(z) * (tf.square((tf.sqrt(1 + 4 * eps * (tf.abs(z) + 1 + eps)) - 1) / (2 * eps)) - 1)

            def h_log(z, eps=1):
                return (tf.sign(z) * tf.log(1. + tf.abs(z)) * eps)

            def h_inv_log(z, eps=1):
                return tf.sign(z) * (tf.math.exp(tf.abs(z) / eps) - 1)

            if self.transformed_bellman:
                transformed = h(self.sil_rewards + self.gamma * h_inv(max_action_values) * (1 - self.sil_terminals))
                targets = transformed
            else:
                targets = self.sil_rewards + (self.gamma * max_action_values * (1 - self.sil_terminals))

            # only update with positive td error, create a mask
            td_error = tf.stop_gradient(targets) - predictions
            mask = tf.where(td_error > 0.0,
                            tf.ones_like(self.sil_rewards),
                            tf.zeros_like(self.sil_rewards))
            self.num_valid_samples = tf.reduce_sum(mask)
            self.num_samples = tf.maximum(self.num_valid_samples,
                                          min_batch_size)
            clipped_td_error = tf.stop_gradient(
                    tf.maximum(tf.zeros_like(td_error), td_error))

            td_loss = tf.losses.huber_loss(
                tf.stop_gradient(targets),
                predictions,
                reduction=tf.losses.Reduction.NONE)

            masked_td_loss = td_loss * mask

            total_loss = tf.reduce_sum(self.sil_imp_weights * masked_td_loss) / self.num_samples

            loss = total_loss

            return td_loss, masked_td_loss, td_error, clipped_td_error, loss

    def grad_cam(self, s_batch, a_batch):
        g = tf.get_default_graph()
        with g.gradient_override_map({'Relu': 'GuidedRelu'}):
            prob = tf.nn.softmax(self.q_value)
            cost = tf.reduce_sum(tf.multiply(self.actions, prob), axis=1)
            # gradient for partial linearization.
            # We only care about target visualization class.
            y_c = tf.reduce_sum(tf.multiply(self.q_value, self.actions), axis=1)

            # Get last convolutional layer gradient for generating gradCAM visualization
            last_conv = self.h_conv3
            last_conv_grad = tf.gradients(y_c, last_conv)[0]
            # Guided backpropagtion back to input layer
            gb_grad = tf.gradients(cost, self.observation)[0]

            conv_value, convgrad_value, gbgrad_value = self.sess.run([last_conv,
                                                last_conv_grad,
                                                gb_grad],
                                                feed_dict={self.observation: s_batch,
                                                           self.actions: a_batch})
            return conv_value, convgrad_value, gbgrad_value

    def train(self, s_j_batch, a_batch, r_batch, s_j1_batch, terminal, importance_weights, global_t):
        if self.target_consistency_loss:
            summary, _, _, td_error = self.sess.run(
                [self.summary_op, self.train_step, self.cost, self.td_error],
                feed_dict={
                    self.observation: s_j_batch,
                    self.tc_observation: s_j1_batch,
                    self.actions: a_batch,
                    self.next_observation: s_j1_batch,
                    self.rewards: r_batch,
                    self.terminals: terminal})
        else:
            summary, _, _, td_error = self.sess.run(
                [self.summary_op, self.train_step, self.cost, self.td_error],
                feed_dict={
                    self.observation: s_j_batch,
                    self.actions: a_batch,
                    self.next_observation: s_j1_batch,
                    self.rewards: r_batch,
                    self.terminals: terminal,
                    self.imp_weights: importance_weights})

        if self.verbose:
            self.add_summary(summary, global_t)

        return td_error


    def sil_train(self, s_j_batch, a_batch, r_batch, s_j1_batch, terminal, importance_weights, global_t):
        _, td_loss, masked_td_loss, sil_td_errpr, clipped_sil_td_error, sil_cost = self.sess.run(
            [self.sil_train_step, self.td_loss, self.masked_td_loss, self.sil_td_error,
             self.clipped_sil_td_error, self.sil_cost],
            feed_dict={
                self.observation: s_j_batch,
                self.sil_actions: a_batch,
                self.next_observation: s_j1_batch,
                self.sil_rewards: r_batch,
                self.sil_terminals: terminal,
                self.sil_imp_weights: importance_weights})

        if self.verbose:
            self.add_summary(summary, global_t)

        return td_loss, masked_td_loss, sil_td_errpr, clipped_sil_td_error, sil_cost


    def get_td_error(self, s_j_batch, a_batch, r_batch, s_j1_batch, terminal, importance_weights, global_t):
        td_error = self.sess.run(
            self.td_error,
            feed_dict={
                self.observation: s_j_batch,
                self.actions: a_batch,
                self.next_observation: s_j1_batch,
                self.rewards: r_batch,
                self.terminals: terminal,
                self.imp_weights: importance_weights})

        return td_error

    def record_summary(self, score=0, steps=0, episodes=None, global_t=0, mode='Test'):
        summary = tf.Summary()
        summary.value.add(tag='{}/score'.format(mode), simple_value=float(score))
        summary.value.add(tag='{}/steps'.format(mode), simple_value=float(steps))
        if episodes is not None:
            summary.value.add(tag='{}/episodes'.format(mode), simple_value=float(episodes))
        self.writer.add_summary(summary, global_t)
        self.writer.flush()

    def add_summary(self, summary, step):
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def load(self, folder=None):
        has_checkpoint = False

        __folder = self.folder
        # saving and loading networks
        if folder is not None:
            __folder = folder
        checkpoint = tf.train.get_checkpoint_state(__folder)

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            logger.debug('checkpoint loaded:{}'.format(checkpoint.model_checkpoint_path))
            sleep(.2)
            tokens = checkpoint.model_checkpoint_path.split("-")
            self.global_t = int(tokens[1])
            has_checkpoint = True

        return has_checkpoint

    def save(self, step=0):
        logger.debug('saving checkpoint...')
        # if step < 0:
        #     self.saver.save(self.sess, self.folder + '/{}_checkpoint_dqn'.format(self.name.replace('-', '_')))
        # else:
        self.saver.save(self.sess, self.folder + '/{}_checkpoint_dqn'.format(self.name.replace('-', '_')), global_step=step)
        logger.info('Successfully saved model checkpoint at step {}!'.format(step))

        # logger.info('Saving convolutional weights as images...')
        # conv_weights = self.sess.run([tf.get_collection('conv_weights')])
        # for i, c in enumerate(conv_weights[0]):
        #     plot_conv_weights(c, 'conv{}'.format(i+1), folder=self.folder)
        # logger.info('Successfully saved convolutional weights!')

    def init_verbosity(self):
        with tf.name_scope("Summary_Conv1") as scope:
            self.variable_summaries(self.W_conv1, 'weights')
            self.variable_summaries(self.b_conv1, 'biases')
            tf.summary.histogram('activations', self.h_conv1)
        with tf.name_scope("Summary_Conv2") as scope:
            self.variable_summaries(self.W_conv2, 'weights')
            self.variable_summaries(self.b_conv2, 'biases')
            tf.summary.histogram('activations', self.h_conv2)
        with tf.name_scope("Summary_Conv3") as scope:
            self.variable_summaries(self.W_conv3, 'weights')
            self.variable_summaries(self.b_conv3, 'biases')
            tf.summary.histogram('/activations', self.h_conv3)
        with tf.name_scope("Summary_Flatten") as scope:
            tf.summary.histogram('activations', self.h_conv3_flat)
        with tf.name_scope("Summary_FullyConnected1") as scope:
            self.variable_summaries(self.W_fc1, 'weights')
            self.variable_summaries(self.b_fc1, 'biases')
            tf.summary.histogram('activations', self.h_fc1)
        with tf.name_scope("Summary_FullyConnected2") as scope:
            self.variable_summaries(self.W_fc2, 'weights')
            self.variable_summaries(self.b_fc2, 'biases')
            tf.summary.histogram('activations', self.action_output)

    def load_transfer_model(self, sess, folder='',
        not_transfer_fc2=False, not_transfer_fc1=False,
        not_transfer_conv3=False, not_transfer_conv2=False):
        assert folder != ''
        assert os.path.isdir(folder)

        logger.info('Initialize network from a pretrain model in {}'.format(folder))

        transfer_all = False
        if not_transfer_conv2:
            folder += '/noconv2'
            var_list = [
                self.W_conv1, self.b_conv1]
        elif not_transfer_conv3:
            folder += '/noconv3'
            var_list = [
                self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2]
        elif not_transfer_fc1:
            folder += '/nofc1'
            var_list = [
                self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_conv3, self.b_conv3]
        elif not_transfer_fc2:
            folder += '/nofc2'
            var_list = [
                self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_conv3, self.b_conv3,
                self.W_fc1, self.b_fc1]
        else:
            transfer_all = True
            with open(folder + "/max_output_value", 'r') as f_max_value:
                transfer_max_output_val = float(f_max_value.readline().split()[0])
            folder += '/all'
            var_list = [
                self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_conv3, self.b_conv3,
                self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2]

        saver_transfer_from = tf.train.Saver(var_list=var_list)
        checkpoint_transfer_from = tf.train.get_checkpoint_state(folder)

        if checkpoint_transfer_from and checkpoint_transfer_from.model_checkpoint_path:
            saver_transfer_from.restore(self.sess, checkpoint_transfer_from.model_checkpoint_path)
            logger.info("Successfully loaded: {}".format(checkpoint_transfer_from.model_checkpoint_path))

            global_vars = tf.global_variables()
            is_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            initialized_vars = [v for (v, f) in zip(global_vars, is_initialized) if f]
            for var in initialized_vars:
                logger.info("    {} loaded".format(var.op.name))
                sleep(1)

            if transfer_all:
                # scale down last layer if it's transferred
                logger.info("Normalizing output layer with max value {}...".format(transfer_max_output_val))
                W_fc2_norm = tf.div(self.W_fc2, transfer_max_output_val)
                b_fc2_norm = tf.div(self.b_fc2, transfer_max_output_val)
                logger.info("Output layer normalized")
                sess.run([
                    self.W_fc2.assign(W_fc2_norm), self.b_fc2.assign(b_fc2_norm)
                ])
