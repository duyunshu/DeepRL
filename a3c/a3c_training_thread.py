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


class A3CTrainingThread(object):
    """Asynchronous Actor-Critic Training Thread Class."""

    log_interval = 100
    perf_log_interval = 1000
    local_t_max = 20
    use_lstm = False
    action_size = -1
    entropy_beta = 0.01
    gamma = 0.99
    use_mnih_2015 = False
    env_id = None
    reward_type = 'CLIP'  # CLIP | LOG | RAW
    finetune_upper_layers_oinly = False
    shaping_reward = 0.001
    shaping_factor = 1.
    shaping_gamma = 0.85
    advice_confidence = 0.8
    shaping_actions = -1  # -1 all actions, 0 exclude noop
    transformed_bellman = False
    clip_norm = 0.5
    use_grad_cam = False
    use_sil = False
    use_sil_neg = False # test if also using samples that are (G-V<0)
    # use_correction = False
    log_idx = 0
    reward_constant = 0

    def __init__(self, thread_index, global_net, local_net,
                 initial_learning_rate, learning_rate_input, grad_applier,
                 max_global_time_step, device=None, pretrained_model=None,
                 pretrained_model_sess=None, advice=False, correction=False,
                 reward_shaping=False, sil_thread=False, classify_thread=False,
                 rollout_thread=False,
                 batch_size=None, no_op_max=30):
        """Initialize A3CTrainingThread class."""
        assert self.action_size != -1

        self.thread_idx = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step
        self.use_pretrained_model_as_advice = advice
        self.use_pretrained_model_as_reward_shaping = reward_shaping
        self.use_correction=correction
        self.local_net = local_net
        self.sil_thread = sil_thread
        if self.sil_thread:
            self.batch_size = batch_size
        self.classify_thread = classify_thread
        self.rollout_thread = rollout_thread

        self.no_op_max = no_op_max
        self.override_num_noops = 0 if self.no_op_max == 0 else None

        logger.info("===A3C thread_index: {}".format(self.thread_idx))
        logger.info("device: {}".format(device))
        logger.info("sil_thread: {}".format(
            colored(self.sil_thread, "green" if self.sil_thread else "red")))
        logger.info("classifier_thread: {}".format(
            colored(self.classify_thread, "green" if self.classify_thread else "red")))
        logger.info("rollout_thread: {}".format(
            colored(self.rollout_thread, "green" if self.rollout_thread else "red")))
        logger.info("use_sil: {}".format(
            colored(self.use_sil, "green" if self.use_sil else "red")))
        logger.info("use_sil_neg: {}".format(
            colored(self.use_sil_neg, "green" if self.use_sil_neg else "red")))
        logger.info("use_correction: {}".format(
            colored(self.use_correction, "green" if self.use_correction else "red")))

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
        logger.info("use_pretrained_model_as_advice: {}".format(
            colored(self.use_pretrained_model_as_advice,
                    "green" if self.use_pretrained_model_as_advice
                    else "red")))
        logger.info("use_pretrained_model_as_reward_shaping: {}".format(
            colored(self.use_pretrained_model_as_reward_shaping,
                    "green" if self.use_pretrained_model_as_reward_shaping
                    else "red")))
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
            self.local_net.prepare_loss(
                entropy_beta=self.entropy_beta, critic_lr=0.5)
            var_refs = [v._ref() for v in local_vars()]

            self.gradients = tf.gradients(
                self.local_net.total_loss, var_refs)

            if self.sil_thread:
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
                self.gradients, grad_norm = tf.clip_by_global_norm(
                    self.gradients, self.clip_norm)
            self.gradients = list(zip(self.gradients, global_vars()))
            self.apply_gradients = grad_applier.apply_gradients(self.gradients)

            if self.sil_thread:
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

        self.pretrained_model = pretrained_model
        self.pretrained_model_sess = pretrained_model_sess
        self.psi = 0.9 if self.use_pretrained_model_as_advice else 0.0
        self.advice_ctr = 0
        self.shaping_ctr = 0
        self.last_rho = 0.

        if self.use_pretrained_model_as_advice \
           or self.use_pretrained_model_as_reward_shaping:
            assert self.pretrained_model is not None

        if self.use_sil:  # and not self.sil_thread:
            self.episode = SILReplayMemory(
                self.action_size, max_len=None, gamma=self.gamma,
                clip=reward_clipped,
                height=self.local_net.in_shape[0],
                width=self.local_net.in_shape[1],
                phi_length=self.local_net.in_shape[2],
                reward_constant=self.reward_constant)

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate \
            * (self.max_global_time_step - global_time_step) \
            / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def pick_action(self, logits):
        """Choose action probabilistically.

        Reference:
        https://github.com/ppyht2/tf-a2c/blob/master/src/policy.py
        """
        noise = np.random.uniform(0, 1, np.shape(logits))
        return np.argmax(logits - np.log(-np.log(noise)))

    def pick_action_w_confidence(self, pi_values, exclude_noop=True):
        """Pick action with confidence."""
        actions_confidence = []
        # exclude NOOP action
        for action in range(1 if exclude_noop else 0, self.action_size):
            actions_confidence.append(pi_values[action][0][0])
        max_confidence_action = np.argmax(actions_confidence)
        confidence = actions_confidence[max_confidence_action]
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

    def set_start_time(self, start_time):
        """Set start time."""
        self.start_time = start_time

    def generate_cam(self, sess, test_cam_si, global_t):
        """Compute Grad-CAM and generate video of Grad-CAM."""
        cam_side_img = []

        for i in range(len(test_cam_si)):
            state = test_cam_si[i]

            if np.isnan(state).any():
                logger.error("Nan in state_img {}".format(i))
                continue

            state = cv2.resize(state, self.local_net.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)

            # get max action per demo state
            readout_t = self.local_net.run_policy(sess, state)
            action = np.argmax(readout_t)

            # convert action to one-hot vector
            action_onehot = [0.] * self.game_state.env.action_space.n
            action_onehot[action] = 1.

            # compute grad cam for conv layer 3
            activations, gradients = self.local_net.evaluate_grad_cam(
                sess, state, action_onehot)
            cam = grad_cam(activations, gradients)
            cam_img = visualize_cam(cam, shape=self.local_net.in_shape[:-1])

            side_by_side = generate_image_for_cam_video(
                state,
                cam_img, global_t, i,
                self.action_meaning[action], shape=self.local_net.in_shape[0])

            cam_side_img.append(side_by_side)

        return cam_side_img

    def generate_cam_video(self, sess, time_per_step, global_t, folder,
                           demo_memory_cam, demo_cam_human=False):
        """Generate the Grad-CAM video."""
        cam_side_img = self.generate_cam(sess, demo_memory_cam, global_t)

        path = 'frames/demo-cam_side_img'
        if demo_cam_human:
            path += '_human'

        make_movie(
            cam_side_img,
            str(folder / '{}{ep:010d}'.format(path, ep=(global_t))),
            duration=len(cam_side_img)*time_per_step,
            true_image=True,
            salience=False)
        del cam_side_img

    def testing_model(self, sess, max_steps, global_t, folder,
                      demo_memory_cam=None, demo_cam_human=False):
        """Test A3C model and generate Grad-CAM."""
        logger.info("Testing model at global_t={}...".format(global_t))
        # copy weights from shared to local
        sess.run(self.sync)

        if demo_memory_cam is not None:
            self.generate_cam_video(sess, 0.03, global_t, folder,
                                    demo_memory_cam, demo_cam_human)
            return
        else:
            self.game_state.reset(hard_reset=True)
            max_steps += 4
            # print(self.game_state.clone_full_state().shape)
            # self.game_state.env.render()
            # int(input())
            test_memory = ReplayMemory(
                84, 84,  # default even if input shape is different
                np.random.RandomState(),
                max_steps=max_steps,
                phi_length=4,
                num_actions=self.game_state.env.action_space.n,
                wrap_memory=False,
                full_state_size=self.game_state.clone_full_state().shape[0])
            for _ in range(4):
                test_memory.add(
                    self.game_state.x_t,
                    0,
                    self.game_state.reward,
                    self.game_state.terminal,
                    self.game_state.lives,
                    fullstate=self.game_state.full_state)

        episode_buffer = []
        test_memory_cam = []

        total_reward = 0
        total_steps = 0
        episode_reward = 0
        episode_steps = 0
        terminal = False
        while True:
            state = cv2.resize(self.game_state.s_t,
                               self.local_net.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)
            test_memory_cam.append(state)
            episode_buffer.append(self.game_state.get_screen_rgb())
            pi_, value_, logits_ = self.local_net.run_policy_and_value(
                sess, state)
            action = np.argmax(pi_)

            self.game_state.step(action)
            terminal = self.game_state.terminal
            memory_full = episode_steps == max_steps-5
            terminal_ = terminal or memory_full

            # store the transition to replay memory
            test_memory.add(
                self.game_state.x_t1, action,
                self.game_state.reward, terminal_,
                self.game_state.lives,
                fullstate=self.game_state.full_state1)

            # update the old values
            episode_reward += self.game_state.reward
            episode_steps += 1

            # s_t = s_t1
            self.game_state.update()

            if terminal_:
                env = self.game_state.env
                name = 'EpisodicLifeEnv'
                if get_wrapper_by_name(env, name).was_real_done or memory_full:
                    time_per_step = 0.03
                    images = np.array(episode_buffer)
                    file = 'frames/image{ep:010d}'.format(ep=global_t)
                    duration = len(images)*time_per_step
                    make_movie(images, str(folder / file), duration=duration,
                               true_image=True, salience=False)
                    break

                self.game_state.reset(hard_reset=False)
                if self.use_lstm:
                    self.local_net.reset_state()

        total_reward = episode_reward
        total_steps = episode_steps
        log_data = (global_t, self.thread_idx, total_reward, total_steps)
        logger.info("test: global_t={} worker={} final score={}"
                    " final steps={}".format(*log_data))

        self.generate_cam_video(sess, 0.03, global_t, folder,
                                np.array(test_memory_cam))
        test_memory.save(name='test_cam', folder=folder, resize=True)

        if self.use_lstm:
            self.local_net.reset_state()

        return

    def testing(self, sess, max_steps, global_t, folder,
        demo_memory_cam=None, worker=None):
        """Evaluate A3C."""
        logger.info("Evaluate policy at global_t={}...".format(global_t))

        # copy weights from shared to local
        sess.run(self.sync)

        if demo_memory_cam is not None and global_t % 5000000 == 0:
            self.generate_cam_video(sess, 0.03, global_t, folder,
                                    demo_memory_cam)

        episode_buffer = []
        self.game_state.reset(hard_reset=True)
        episode_buffer.append(self.game_state.get_screen_rgb())

        total_reward = 0
        total_steps = 0
        episode_reward = 0
        episode_steps = 0
        n_episodes = 0
        while max_steps > 0:
            state = cv2.resize(self.game_state.s_t,
                               self.local_net.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)
            pi_, value_, logits_ = self.local_net.run_policy_and_value(
                sess, state)

            if False:
                action = np.random.choice(range(self.action_size), p=pi_)
            else:
                action = self.pick_action(logits_)

            if self.use_pretrained_model_as_advice:
                psi = self.psi if self.psi > 0.001 else 0.0

                if psi > np.random.rand():
                    # TODO(add shape as attribute to pretrained model, fix s_t)
                    model_pi = self.pretrained_model.run_policy(
                        self.pretrained_model_sess, self.game_state.s_t)
                    model_action, confidence = self.pick_action_w_confidence(
                        model_pi, exclude_noop=False)

                    if model_action > self.shaping_actions \
                       and confidence >= self.advice_confidence:
                        action = model_action

            # take action
            self.game_state.step(action)
            terminal = self.game_state.terminal

            if n_episodes == 0 and global_t % 5000000 == 0:
                episode_buffer.append(self.game_state.get_screen_rgb())

            episode_reward += self.game_state.reward
            episode_steps += 1
            max_steps -= 1

            # s_t = s_t1
            self.game_state.update()

            if terminal:
                env = self.game_state.env
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
                    log_data = (global_t, self.thread_idx, n_episodes,
                                score_str, steps_str, total_steps)
                    logger.debug("test: global_t={} worker={} trial={} {} {}"
                                 " total_steps={}".format(*log_data))
                    total_reward += episode_reward
                    total_steps += episode_steps
                    episode_reward = 0
                    episode_steps = 0

                self.game_state.reset(hard_reset=False)
                if self.use_lstm:
                    self.local_net.reset_state()

        if n_episodes == 0:
            total_reward = episode_reward
            total_steps = episode_steps
        else:
            total_reward = total_reward / n_episodes
            total_steps = total_steps // n_episodes

        log_data = (global_t, self.thread_idx, total_reward, total_steps,
                    n_episodes)
        logger.info("test: global_t={} worker={} final score={} final steps={}"
                    " # trials={}".format(*log_data))

        self.record_summary(
            score=total_reward, steps=total_steps,
            episodes=n_episodes, global_t=global_t, mode='A3C_Test')

        # reset variables used in training
        self.episode_reward = 0
        self.episode_steps = 0
        self.game_state.reset(hard_reset=True)
        self.last_rho = 0.

        if self.use_lstm:
            self.local_net.reset_state()

        if self.use_sil and not self.sil_thread:
            # ensure no states left from a non-terminating episode
            self.episode.reset()
        return (total_reward, total_steps, n_episodes)

    def train(self, sess, global_t, train_rewards):
        """Train A3C."""
        states = []
        fullstates=[]
        actions = []
        rewards = []
        values = []
        rho = []

        terminal_pseudo = False  # loss of life
        terminal_end = False  # real terminal

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t

        if self.use_lstm:
            start_lstm_state = self.local_net.lstm_state_out

        # t_max times loop
        for i in range(self.local_t_max):
            state = cv2.resize(self.game_state.s_t,
                               self.local_net.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)
            fullstate = self.game_state.clone_full_state()

            pi_, value_, logits_ = self.local_net.run_policy_and_value(sess,
                                                                       state)
            action = self.pick_action(logits_)

            model_pi = None
            confidence = 0.
            if self.use_pretrained_model_as_advice:
                assert self.pretrained_model is not None

                if self.psi > 0.001:
                    self.psi = 0.9999 * (0.9999 ** global_t)  # 0.99995 works
                else:
                    self.psi = 0.0

                if self.psi > np.random.rand():
                    # TODO(add shape as attribute to pretrained model, fix s_t)
                    model_pi = self.pretrained_model.run_policy(
                        self.pretrained_model_sess, self.game_state.s_t)
                    model_action, confidence = self.pick_action_w_confidence(
                        model_pi, exclude_noop=False)

                    if (model_action > self.shaping_actions
                       and confidence >= self.advice_confidence):
                        action = model_action
                        self.advice_ctr += 1

            if self.use_pretrained_model_as_reward_shaping:
                assert self.pretrained_model is not None
                # if action > 0:
                if model_pi is None:
                    # TODO(add shape as attribute to pretrained model, fix s_t)
                    model_pi = self.pretrained_model.run_policy(
                        self.pretrained_model_sess, self.game_state.s_t)
                    confidence = model_pi[action][0][0]

                if (action > self.shaping_actions
                   and confidence >= self.advice_confidence):
                    # rho.append(round(confidence, 5))
                    rho.append(self.shaping_reward)
                    self.shaping_ctr += 1
                else:
                    rho.append(0.)
                # self.shaping_ctr += 1

            states.append(state)
            fullstates.append(fullstate)  # make sure the idx of states and fullstates are the same
            actions.append(action)
            values.append(value_)

            if self.thread_idx == self.log_idx \
               and self.local_t % self.log_interval == 0:
                log_msg1 = "lg={}".format(np.array_str(
                    logits_, precision=4, suppress_small=True))
                log_msg2 = "pi={}".format(np.array_str(
                    pi_, precision=4, suppress_small=True))
                log_msg3 = "V={:.4f}".format(value_)

                if self.use_pretrained_model_as_advice:
                    log_msg3 += " psi={:.4f}".format(self.psi)

                logger.debug(log_msg1)
                logger.debug(log_msg2)
                logger.debug(log_msg3)

            # process game
            self.game_state.step(action)

            # receive game result
            reward = self.game_state.reward
            terminal = self.game_state.terminal
            if self.use_pretrained_model_as_reward_shaping:
                if reward < 0 and reward > 0:
                    rho[i] = 0.
                    j = i-1
                    while j > i-5:
                        if rewards[j] != 0:
                            break
                        rho[j] = 0.
                        j -= 1

            self.episode_reward += reward

            if self.use_sil:
                # save states in episode memory
                self.episode.add_item(self.game_state.s_t, fullstate,
                                      action, reward, terminal)

            if self.reward_type == 'LOG':
                reward = np.sign(reward) * np.log(1 + np.abs(reward))
            elif self.reward_type == 'CLIP':
                reward = np.sign(reward)

            rewards.append(reward)

            self.local_t += 1
            self.episode_steps += 1
            global_t += 1

            # s_t1 -> s_t
            self.game_state.update()

            if terminal:
                terminal_pseudo = True

                env = self.game_state.env
                name = 'EpisodicLifeEnv'
                if get_wrapper_by_name(env, name).was_real_done:
                    log_msg = "train: worker={} global_t={} local_t={}".format(
                        self.thread_idx, global_t, self.local_t)

                    if self.use_pretrained_model_as_advice:
                        log_msg += " advice_ctr={}".format(self.advice_ctr)
                    if self.use_pretrained_model_as_reward_shaping:
                        log_msg += " shaping_ctr={}".format(self.shaping_ctr)

                    score_str = colored("score={}".format(
                        self.episode_reward), "magenta")
                    steps_str = colored("steps={}".format(
                        self.episode_steps), "blue")
                    log_msg += " {} {}".format(score_str, steps_str)
                    logger.debug(log_msg)
                    train_rewards['train'][global_t] = (self.episode_reward,
                                                        self.episode_steps)
                    self.record_summary(
                        score=self.episode_reward, steps=self.episode_steps,
                        episodes=None, global_t=global_t, mode='Train')
                    self.episode_reward = 0
                    self.episode_steps = 0
                    terminal_end = True

                self.last_rho = 0.
                if self.use_lstm:
                    self.local_net.reset_state()
                self.game_state.reset(hard_reset=False)
                break

        cumsum_reward = 0.0
        if not terminal:
            state = cv2.resize(self.game_state.s_t,
                               self.local_net.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)
            cumsum_reward = self.local_net.run_value(sess, state)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_state = []
        batch_action = []
        batch_adv = []
        batch_cumsum_reward = []

        if self.use_pretrained_model_as_reward_shaping:
            rho.reverse()
            rho.append(self.last_rho)
            self.last_rho = rho[0]
            i = 0
            # compute and accumulate gradients
            for(ai, ri, si, vi) in zip(actions, rewards, states, values):
                # Wiewiora et al.(2003) Principled Methods for Advising RL
                # Look-Back Advice
                # F = rho[i] - (self.shaping_gamma**-1) * rho[i+1]
                # F = rho[i] - self.shaping_gamma * rho[i+1]
                f = (self.shaping_gamma**-1) * rho[i] - rho[i+1]
                if (i == 0 and terminal) or (f != 0 and (ri > 0 or ri < 0)):
                    # logger.warn("averted additional F in absorbing state")
                    f = 0.
                # if (F < 0. and ri > 0) or (F > 0. and ri < 0):
                #     logger.warn("Negative reward shaping F={} ri={}"
                #                 " rho[s]={} rhos[s-1]={}".format(
                #                 F, ri, rho[i], rho[i+1]))
                #     F = 0.
                cumsum_reward = (ri + f*self.shaping_factor) \
                    + self.gamma * cumsum_reward
                advantage = cumsum_reward - vi

                a = np.zeros([self.action_size])
                a[ai] = 1

                batch_state.append(si)
                batch_action.append(a)
                batch_adv.append(advantage)
                batch_cumsum_reward.append(cumsum_reward)
                i += 1
        else:
            # compute and accumulate gradients
            for(ai, ri, si, vi) in zip(actions, rewards, states, values):
                if self.transformed_bellman:
                    ri = np.sign(ri) * 1.89 + ri
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

        cur_learning_rate = self._anneal_learning_rate(global_t)

        if self.use_lstm:
            batch_state.reverse()
            batch_action.reverse()
            batch_adv.reverse()
            batch_cumsum_reward.reverse()
            feed_dict = {
                self.local_net.s: batch_state,
                self.local_net.a: batch_action,
                self.local_net.advantage: batch_adv,
                self.local_net.cumulative_reward: batch_cumsum_reward,
                self.local_net.initial_lstm_state: start_lstm_state,
                self.local_net.step_size: [len(batch_action)],
                self.learning_rate_input: cur_learning_rate,
                }
        else:
            feed_dict = {
                self.local_net.s: batch_state,
                self.local_net.a: batch_action,
                self.local_net.advantage: batch_adv,
                self.local_net.cumulative_reward: batch_cumsum_reward,
                self.learning_rate_input: cur_learning_rate,
                }

        sess.run(self.apply_gradients, feed_dict=feed_dict)

        t = self.local_t - self.prev_local_t
        if (self.thread_idx == self.log_idx and t >= self.perf_log_interval):
            self.prev_local_t += self.perf_log_interval
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            logger.info("worker-{}, log_worker-{}".format(self.thread_idx, self.log_idx))
            logger.info("Performance : {} STEPS in {:.0f} sec. {:.0f}"
                        " STEPS/sec. {:.2f}M STEPS/hour".format(
                            global_t,  elapsed_time, steps_per_sec,
                            steps_per_sec * 3600 / 1000000.))

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t, terminal_end, terminal_pseudo

    def sil_train(self, sess, global_t, sil_memory, sil_ctr, m=4):
        """Self-imitation learning process."""
        assert not self.use_lstm

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
        assert not self.use_lstm

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


    def test_game_classifier(self, global_t, max_steps, sess, worker=None):
        """Evaluate game with current classifier model."""
        if not self.classify_thread:
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
                model_pi, exclude_noop=False)

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
        # return log_data
