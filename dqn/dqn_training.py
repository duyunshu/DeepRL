#!/usr/bin/env python3
import tensorflow as tf
import cv2
import sys
import os
import random
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from skimage.transform import resize
import matplotlib.image as mpimg

from termcolor import colored
from common.util import egreedy, get_action_index, make_movie, load_memory, compute_returns
from common.game_state import get_wrapper_by_name
from setup_functions import test

logger = logging.getLogger("dqn")

ACTION_MEANING = {
    0 : "NOOP",
    1 : "FIRE",
    2 : "UP",
    3 : "RIGHT",
    4 : "LEFT",
    5 : "DOWN",
    6 : "UPRIGHT",
    7 : "UPLEFT",
    8 : "DOWNRIGHT",
    9 : "DOWNLEFT",
    10 : "UPFIRE",
    11 : "RIGHTFIRE",
    12 : "LEFTFIRE",
    13 : "DOWNFIRE",
    14 : "UPRIGHTFIRE",
    15 : "UPLEFTFIRE",
    16 : "DOWNRIGHTFIRE",
    17 : "DOWNLEFTFIRE",
}

try:
    import cPickle as pickle
except ImportError:
    import pickle

class DQNTraining(object):
    def __init__(
        self, sess, network, game_state, eval_game_state, resized_height, resized_width,
        phi_length, n_actions, batch, name, gamma, observe, explore,
        final_epsilon, init_epsilon, replay_buffer,
        update_freq, save_freq, eval_freq, eval_max_steps, copy_freq,
        folder, load_demo_memory=False, demo_memory_folder=None, demo_ids=None,
        load_demo_cam=False, demo_cam_id=None,
        train_max_steps=sys.maxsize, human_net=None, confidence=0., psi=0.999995,
        train_with_demo_steps=0, use_transfer=False, reward_type='CLIP',
        priority_memory=False, beta_schedule=None, prioritized_replay_eps=1e-6,
        prio_by_td=False, prio_by_return=False, update_q_with_return=False,
        use_rollout=False, rollout_buffer=None, rollout_worker=None, temp_buffer=None,
        update_in_rollout=False, use_sil=False, sil_buffer=None, sil_priority_memory=False):
        """ Initialize experiment """
        self.sess = sess
        self.net = network
        self.game_state = game_state
        self.eval_game_state = eval_game_state
        self.observe = observe
        self.explore = explore
        self.final_epsilon = final_epsilon
        self.init_epsilon = init_epsilon
        self.update_freq = update_freq # backpropagate frequency
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.eval_max_steps = eval_max_steps
        self.copy_freq = copy_freq # copy q to t-network frequency
        self.resized_h = resized_height
        self.resized_w = resized_width
        self.phi_length = phi_length
        self.n_actions = n_actions
        self.batch = batch
        self.name = name
        self.folder = folder
        self.load_demo_memory = load_demo_memory
        self.demo_memory_folder = demo_memory_folder
        self.demo_ids = demo_ids
        self.load_demo_cam = load_demo_cam
        self.demo_cam_id = demo_cam_id
        self.train_max_steps = train_max_steps
        self.train_with_demo_steps = train_with_demo_steps
        self.use_transfer = use_transfer
        self.reward_type = reward_type

        self.priority_memory = priority_memory
        self.prioritized_replay_eps=prioritized_replay_eps
        self.prio_by_td = prio_by_td
        self.prio_by_return = prio_by_return
        assert not (self.prio_by_td and self.prio_by_return), "can't prio by both, must pick either td OR return"
        self.update_q_with_return = update_q_with_return

        self.human_net = human_net
        self.confidence = confidence
        self.use_human_advice = False
        self.psi = self.init_psi = psi
        if self.human_net is not None:
            self.use_human_advice = True

        self.replay_buffer = replay_buffer

        self.use_rollout = use_rollout
        self.rollout_buffer = rollout_buffer
        self.rollout_worker = rollout_worker
        self.beta_schedule = beta_schedule
        self.update_in_rollout = update_in_rollout

        self.use_sil = use_sil
        self.sil_buffer = sil_buffer
        self.sil_priority_memory = sil_priority_memory

        self.temp_buffer = temp_buffer

        if not os.path.exists(self.folder + '/frames'):
            os.makedirs(self.folder + '/frames')

        logger.info("===DQN thread===")#.format(self.thread_idx))
        logger.info("observe before start learning: {}".format(self.observe))
        logger.info("exploration period: {}".format(self.explore))
        logger.info("init epsilon: {}, final epsilon: {}".format(self.init_epsilon, self.final_epsilon))
        logger.info("copy target net period: {}".format(self.copy_freq))
        logger.info("update frequency: {}".format(self.update_freq))
        logger.info("action_size: {}".format(self.n_actions))
        logger.info("reward_type: {}".format(self.reward_type))
        logger.info("priority memory: {}".format(
            colored(self.priority_memory,
                    "green" if self.priority_memory else "red")))
        logger.info("use sil: {}".format(
            colored(self.use_sil,
                    "green" if self.use_sil else "red")))
        logger.info("SIL priority memory: {}".format(
            colored(self.sil_priority_memory,
                    "green" if self.sil_priority_memory else "red")))
        logger.info("use rollout: {}".format(
            colored(self.use_rollout,
                    "green" if self.use_rollout else "red")))
        time.sleep(2)

    def _reset(self, hard_reset=True):
        self.game_state.reset(hard_reset=hard_reset)

    def _add_demo_experiences(self):
        assert self.demo_memory_folder is not None
        demo_memory, actions_ctr, total_rewards, total_steps = load_memory(
            name=None,
            demo_memory_folder=self.demo_memory_folder,
            demo_ids=self.demo_ids,
            imgs_normalized=False)

        logger.info("Memory size={}".format(self.replay_buffer.size))
        logger.info("Adding human experiences...")
        for idx in list(demo_memory.keys()):
            demo = demo_memory[idx]
            for i in range(demo.max_steps):
                self.replay_buffer.add(
                    demo.imgs[i], demo.actions[i],
                    demo.rewards[i], demo.terminal[i],
                    demo.lives[i], demo.full_state[i])
            demo.close()
            del demo
        logger.info("Memory size={}".format(self.replay_buffer.size))
        time.sleep(2)

    def _load(self):
        if self.net.load():
            # set global step
            self.global_t = self.net.global_t
            logger.info(">>> global step set: {}".format(self.global_t))
            # set wall time
            wall_t_fname = self.folder + '/' + 'wall_t.' + str(self.global_t)
            with open(wall_t_fname, 'r') as f:
                wall_t = float(f.read())
            # set epsilon
            epsilon_fname = self.folder + '/epsilon'
            with open(epsilon_fname, 'r') as f:
                self.epsilon = float(f.read())
            self.rewards = pickle.load(open(self.folder + '/' + self.name.replace('-', '_') + '-dqn-rewards.pkl', 'rb'))
            self.replay_buffer.load(name=self.name, folder=self.folder)
        else:
            logger.warn("Could not find old network weights")
            if self.load_demo_memory:
                self._add_demo_experiences()
            self.global_t = 0
            self.epsilon = self.init_epsilon
            self.rewards = {'train':{}, 'eval':{}}
            wall_t = 0.0
        return wall_t

    def visualize(self, conv_output, conv_grad):  # image, gb_viz):
        output = conv_output
        grads_val = conv_grad

        # global average pooling
        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.zeros(output.shape[0:2], dtype=np.float32)
        # cam = np.ones(output.shape[0:2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * output[:,:,i]
        # passing through Relu
        cam = np.maximum(cam, 0) # only care about positive
        cam = cam / np.max(cam) # scale to [0,1]
        cam = resize(cam, (84, 84), preserve_range=True)
        cam_heatmap = cv2.applyColorMap(np.uint8(225*cam), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

        return cam_heatmap

    def calculate_cam(self, test_cam_si):
        state = []
        action_onehot = []
        action_array = []

        for i in range(len(test_cam_si)):
            readout_t = self.net.evaluate(test_cam_si[i])[0]
            action = get_action_index(readout_t,
                is_random=(random.random() <= 0.05),
                n_actions=self.n_actions)
            action_array.append(action)
            a_onehot = np.zeros(self.n_actions)
            a_onehot[action] = 1
            action_onehot.append(a_onehot)

            state.append(np.mean(test_cam_si[i], axis=-1))

        conv_value, conv_grad, gbgrad = self.net.grad_cam(test_cam_si,
                                                          action_onehot)
        cam = []
        img = []

        for i in range(len(conv_value)):
            cam_tmp = self.visualize(conv_value[i], conv_grad[i])
            cam.append(cam_tmp)

            # fake RGB channels for demo images
            state_tmp = cv2.merge((state[i], state[i], state[i]))
            img.append(state_tmp)

        return np.array(cam), np.array(img), action_array

    def train_with_demo_memory_only(self):
        assert self.load_demo_memory
        logger.info((colored('Training with demo memory only for {} steps...'.format(self.train_with_demo_steps), 'blue')))
        start_update_counter = self.net.update_counter
        while self.train_with_demo_steps > 0:
            if self.use_transfer:
                self.net.update_counter = 1 # this ensures target network doesn't update
            s_j_batch, a_batch, r_batch, s_j1_batch, terminals = self.replay_buffer.random_batch(self.batch)
            # perform gradient step
            self.net.train(s_j_batch, a_batch, r_batch, s_j1_batch, terminals)
            self.train_with_demo_steps -= 1
            if self.train_with_demo_steps % 10000 == 0:
                logger.info("\t{} train with demo steps left".format(self.train_with_demo_steps))
        self.net.update_counter = start_update_counter
        self.net.update_target_network(slow=self.net.slow)
        logger.info((colored('Training with demo memory only completed!', 'green')))

    def run(self):
        # load if starting from a checkpoint
        wall_t = self._load()

        # get the first state by doing nothing and preprocess the image to 80x80x4
        # only reset when it doesn't evaluate first when it enters loop below
        if self.global_t % self.eval_freq != 0:
            self._reset(hard_reset=True)

        # only executed at the very beginning of training and never again
        if self.global_t == 0 and self.train_with_demo_steps > 0:
            self.train_with_demo_memory_only()

        # load one demo for cam
        if self.load_demo_cam:
            # note, tuple length has to be >=2. pad 0 if len==1
            demo_cam_id = tuple(map(int, self.demo_cam_id.split(",")))
            if len(demo_cam_id) == 1:
                demo_cam_id = (*demo_cam_id, '0')
            demo_cam, _, total_rewards_cam, _ = load_memory(
                name=None,
                demo_memory_folder=self.demo_memory_folder,
                demo_ids=demo_cam_id,
                imgs_normalized=False)

            max_idx, _ = max(total_rewards_cam.items(), key=lambda a: a[1])
            size_max_idx_mem = len(demo_cam[max_idx])
            self.test_cam_si = np.zeros(
                (size_max_idx_mem,
                 demo_cam[max_idx].height,
                 demo_cam[max_idx].width,
                 demo_cam[max_idx].phi_length),
                dtype=np.float32)
            for i in range(size_max_idx_mem):
                s0, _, _, _, _, _, _, _ = demo_cam[max_idx][i]
                self.test_cam_si[i] = np.copy(s0)
            logger.info("loaded demo {} for testing CAM".format(demo_cam_id))

        # set start time
        start_time = time.time() - wall_t

        logger.info("replay memory size={}, capacity: {}".format(self.replay_buffer.__len__(), self.replay_buffer._maxsize))
        if self.use_rollout:
            assert self.rollout_buffer is not None
            logger.info("refresh memory size={}, capacity: {}".format(self.rollout_buffer.__len__(), self.replay_buffer._maxsize))
        if self.use_sil:
            assert self.sil_buffer is not None
            logger.info("sil memory size={}, capacity: {}".format(self.sil_buffer.__len__(), self.sil_buffer._maxsize))

        sub_total_reward = 0.0
        sub_steps = 0

        rollout_ctr, added_rollout_ctr = 0, 0 # TODO add to parameters

        num_policy_update = 0
        num_agent_update = 0
        num_rollout_update = 0
        num_sil_update = 0

        # init temp array
        states = []
        fullstates = []
        actions = []
        rewards = []
        terminals = []
        next_states = []

        copy_target_freq = self.copy_freq // self.update_freq
        next_save_t = 1

        while self.global_t < self.train_max_steps:
            # agent perform one trajectory, update policy every 4 steps
            while True:
                # Evaluation of policy
                if self.global_t % self.eval_freq == 0:
                    terminal = 0
                    total_reward, total_steps, n_episodes, self.rewards = \
                        test(self.global_t, self.eval_game_state,
                             self.eval_max_steps, self.net, self.n_actions,
                             self.folder, self.rewards)
                    # # re-initialize game for training
                    # self._reset(hard_reset=True)
                    # sub_total_reward = 0.0
                    # sub_steps = 0
                    # time.sleep(0.5)

                #if self.global_t % self.copy_freq == 0:
                # sync target network based on number of policy updates (not number of steps)
                if num_policy_update != 0 and num_policy_update % copy_target_freq == 0:
                    self.net.update_target_network(slow=False)

                # choose an action epsilon greedily
                ## self._update_state_input(observation)
                state = self.game_state.s_t
                fullstate = self.game_state.clone_full_state()
                readout_t = self.net.evaluate(self.game_state.s_t)[0]
                action = get_action_index(
                    readout_t,
                    is_random=(random.random() <= self.epsilon or self.global_t <= self.observe),
                    n_actions=self.n_actions)

                states.append(state)
                fullstates.append(fullstate)
                actions.append(action)

                # scale down epsilon
                if self.epsilon > self.final_epsilon: #and self.global_t > self.observe:
                    self.epsilon -= (self.init_epsilon - self.final_epsilon) / self.explore

                # run the selected action and observe next state and reward
                next_state, reward, terminal = self.game_state.step(action)
                # # terminal = self.game_state.terminal
                # terminal_ = terminal #or ((self.global_t+1) % self.eval_freq == 0)

                sub_total_reward += reward

                if self.reward_type == 'LOG':
                    reward = np.sign(reward) * np.log(1 + np.abs(reward))
                elif self.reward_type == 'CLIP':
                    reward = np.sign(reward)
                elif self.reward_type == 'RAW':
                    reward = reward

                rewards.append(reward)
                terminals.append(terminal)
                next_states.append(next_state)

                # update the old values
                sub_steps += 1
                self.global_t += 1
                self.game_state.update()

                # only train if done random explore (observing); train every 4 steps
                if self.global_t > self.observe and self.global_t % self.update_freq == 0:
                    actions_onehot = []
                    if self.priority_memory: # priority mem
                        experience = self.replay_buffer.sample(self.batch,
                                                               beta=self.beta_schedule.value(self.global_t))
                        s_batch, _, a_batch, r_batch, s1_batch, t_batch, \
                            G_batch, _, weights, idxes = experience
                    else: # non priority mem
                        experience = self.replay_buffer.sample(self.batch)
                        s_batch, _, a_batch, r_batch, s1_batch, t_batch, \
                            G_batch, _ = experience
                        weights, idxes = np.ones_like(r_batch), None

                    # convert action to one-hot vector
                    for ai in a_batch:
                        a = np.zeros([self.n_actions])
                        a[ai] = 1
                        actions_onehot.append(a)
                    td_errors = self.net.train(s_batch, actions_onehot, r_batch,
                                               s1_batch, t_batch, weights, self.global_t)
                    num_policy_update += 1
                    num_agent_update += 1

                    # update priority by TD error (default)
                    if self.priority_memory:
                        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                        self.replay_buffer.update_priorities(idxes, new_priorities)

                if terminal:
                    if get_wrapper_by_name(self.game_state.env, 'EpisodicLifeEnv').was_real_done:
                        self.rewards['train'][self.global_t] = (sub_total_reward, sub_steps)
                        score_str = colored("score={}".format(sub_total_reward), "magenta")
                        steps_str = colored("steps={}".format(sub_steps), "blue")
                        log_data = (self.global_t, score_str, steps_str, num_agent_update)
                        logger.debug("train: global_t={} {} {} num_agent_updates={}".format(*log_data))
                        if self.use_sil:
                            logger.debug("num_sil_updates={}".format(num_sil_update))
                        if self.use_rollout:
                            logger.debug("num_rollout_updates={}".format(num_rollout_update))
                        self.net.record_summary(
                            score=sub_total_reward, steps=sub_steps,
                            episodes=None, global_t=self.global_t, mode='Train')
                        sub_total_reward = 0.0
                        sub_steps = 0

                    # store the transition in D, need to wait till termina & returns are computed
                    # potential problem: add to buffer is slower,
                    # when performing update, might not be using the newest data
                    # combine CER (zhang&sutton 2017)? when making an update, must use the most recent one
                    # i.e., sample 31 from buffer, plus 1 most recent
                    reward_clipped = True if self.reward_type == "CLIP" else False
                    returns = compute_returns(rewards, terminals,
                                              reward_clipped=reward_clipped)
                    for (s, fs, a, r, ns, t, ret) in \
                        zip(states, fullstates, actions, rewards, next_states, terminals, returns):
                        self.replay_buffer.add(obs_t=s, fs=fs, action=a, R=r,
                                               obs_t_next=ns, done=t, returns=ret,
                                               from_rollout=False)
                        if self.use_sil: # also add to sil buffer if using sil
                            self.sil_buffer.add(obs_t=s, fs=fs, action=a, R=r,
                                                obs_t_next=ns, done=t, returns=ret,
                                                from_rollout=False)
                    # reset temp array
                    states = []
                    fullstates = []
                    actions = []
                    rewards = []
                    terminals = []
                    next_states = []

                    # reset env
                    self._reset(hard_reset=False)
                    break

            # AUG24 adding real SIL process
            # this will be sampling from self.replay_buffer and update with (R-Q)+
            if self.use_sil and self.global_t > self.observe:
                for _ in range(4): # use 4 as in SIL
                    actions_onehot = []
                    if self.sil_priority_memory: # priority mem
                        experience = self.sil_buffer.sample(self.batch, beta=0.4) # fixed beta as in SIL
                        s_batch, _, a_batch, r_batch, s1_batch, t_batch, \
                            G_batch, _, weights, idxes = experience
                    else: # non priority mem
                        experience = self.sil_buffer.sample(self.batch)
                        s_batch, _, a_batch, r_batch, s1_batch, t_batch, \
                            G_batch, _ = experience
                        weights, idxes = np.ones_like(r_batch), None

                    # convert action to one-hot vector
                    for ai in a_batch:
                        a = np.zeros([self.n_actions])
                        a[ai] = 1
                        actions_onehot.append(a)
                    # perform gradient step
                    if self.update_q_with_return:
                        td_loss, masked_td_loss, sil_td_errpr, clipped_sil_td_error, sil_cost = \
                            self.net.sil_train(s_batch, actions_onehot, G_batch,
                                               s1_batch, t_batch, weights, self.global_t)
                    else:
                        td_loss, masked_td_loss, sil_td_errpr, clipped_sil_td_error, sil_cost = \
                            self.net.sil_train(s_batch, actions_onehot, r_batch,
                                               s1_batch, t_batch, weights, self.global_t)
                    # update priority by (R-Q)+
                    if self.sil_priority_memory:
                        new_priorities = clipped_sil_td_error #+ self.prioritized_replay_eps # no prio constant as in SIL
                        self.sil_buffer.update_priorities(idxes, new_priorities)

                    num_sil_update += 1
                    num_policy_update += 1
                    if num_sil_update % 100 == 0:
                        logger.info("num_sil_updates:" + str(num_sil_update))


            # entering rollout loop, refresh past states for one episode
            if self.use_rollout and self.global_t > self.observe:
                sample = None
                if self.priority_memory:
                    sample = self.replay_buffer.sample_one_random(1)
                else:
                    sample = self.replay_buffer.sample(1)

                train_out = self.rollout_worker.rollout(self.sess, self.net,
                                    global_t=self.global_t,
                                    samplestate=sample,
                                    rollout_ctr=rollout_ctr,
                                    added_rollout_ctr=added_rollout_ctr,
                                    add_all_rollout=False,
                                    eval_freq=self.eval_freq,
                                    reward_dict=self.rewards,
                                    eval_game_state=self.eval_game_state,
                                    eval_max_steps=self.eval_max_steps,
                                    n_actions=self.n_actions,
                                    folder=self.folder)
                local_t, diff_t, rollout_ctr, added_rollout_ctr, data, self.rewards = train_out

                # add good data to rollout buffer
                if data is not None:
                    ss, ffs, aa, rr, nnss, tt, rett = data
                    # update immedieatly
                    if self.update_in_rollout:
                        # convert action to one-hot vector
                        actions_onehot = []
                        for ai in aa:
                            a = np.zeros([self.n_actions])
                            a[ai] = 1
                            actions_onehot.append(a)
                        # # perform gradient step
                        # if self.update_q_with_return:
                        #     td_errors = self.net.train(ss, actions_onehot, rett,
                        #                                nnss, tt, np.ones_like(rr), self.global_t)
                        # else:
                        td_errors = self.net.train(ss, actions_onehot, rr,
                                                   nnss, tt, np.ones_like(rr), self.global_t)
                        # e.g., if batch size=32, and used 64 data here, update should increase by 2
                        num_policy_update += int(len(rr) / self.batch)
                        num_rollout_update += int(len(rr) / self.batch)

                    # then add to rollout_buffer
                    for (s, fs, a, r, ns, t, ret) in zip(ss, ffs, aa, rr, nnss, tt, rett):
                        self.rollout_buffer.add(obs_t=s, fs=fs, action=a, R=r,
                                                obs_t_next=ns, done=t, returns=ret,
                                                from_rollout=True)

                # makeup for the policy update during rollout
                # make sure there is 1 policy update per 4 global_t
                # perform refresher update with mixed data
                mod = diff_t // self.update_freq

                # update global_t
                self.global_t += diff_t
                for _ in range(mod):
                    if num_policy_update % copy_target_freq == 0:
                        self.net.update_target_network(slow=False)

                    # sample from agent data
                    D_data, D_weights, D_idx = \
                        self.rollout_worker.mixed_sampling(self.priority_memory,
                                                        self.replay_buffer,
                                                        self.batch, self.beta_schedule,
                                                        self.global_t)
                    # sample from rollout data
                    if self.rollout_buffer.__len__() > self.batch:
                        R_data, R_weights, R_idx = \
                            self.rollout_worker.mixed_sampling(self.priority_memory,
                                                            self.rollout_buffer,
                                                            self.batch, self.beta_schedule,
                                                            self.global_t)
                    # add to temp buffer and overwrite priority
                    if self.priority_memory:
                        self.temp_buffer.add_batch_priority(D_data, D_weights)
                        if self.rollout_buffer.__len__() > self.batch:
                            self.temp_buffer.add_batch_priority(R_data, R_weights)
                    else:
                        self.temp_buffer.add_batch(D_data)
                        if self.rollout_buffer.__len__() > self.batch:
                            self.temp_buffer.add_batch(R_data)
                    # mixed sampling
                    data, weights, _ = \
                        self.rollout_worker.mixed_sampling(self.priority_memory,
                                                        self.temp_buffer,
                                                        self.batch,
                                                        self.beta_schedule,
                                                        self.global_t)
                    # update policy
                    s_batch, _, a_batch, r_batch, s1_batch, t_batch, G_batch, _ = data
                    # convert action to one-hot vector
                    actions_onehot = []
                    for ai in a_batch:
                        a = np.zeros([self.n_actions])
                        a[ai] = 1
                        actions_onehot.append(a)
                    # perform gradient step
                    if self.update_q_with_return:
                        td_loss, masked_td_loss, sil_td_errpr, clipped_sil_td_error, sil_cost = \
                            self.net.sil_train(s_batch, actions_onehot, G_batch,
                                               s1_batch, t_batch, weights, self.global_t)
                    else:
                        td_loss, masked_td_loss, sil_td_errpr, clipped_sil_td_error, sil_cost = \
                            self.net.sil_train(s_batch, actions_onehot, r_batch,
                                               s1_batch, t_batch, weights, self.global_t)
                    # print("td_loss:{}\n masked_td_loss:{}\n sil_td_error:{}\n "
                    #       "clipped_sil_td_error:{}\n sil_cost:{}".format(td_loss, masked_td_loss,
                    #                                                      sil_td_errpr, clipped_sil_td_error,
                    #                                                      sil_cost))
                    # int(input())
                    num_policy_update += 1
                    num_rollout_update += 1

                    # update priority
                    if self.priority_memory:
                        if self.prio_by_td:
                            td_errors = self.rollout_worker.compute_priority(self.net, D_data, D_weights, self.global_t)
                            new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                            self.replay_buffer.update_priorities(D_idx, new_priorities)
                            if self.rollout_buffer.__len__() > self.batch:
                                td_errors = self.rollout_worker.compute_priority(self.net, R_data, R_weights, self.global_t)
                                new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                                self.rollout_buffer.update_priorities(R_idx, new_priorities)
                        elif self.prio_by_return:
                            raise Exception("NotImplemented!")


            # save progress every SAVE_FREQ iterations
            # if self.global_t % self.save_freq == 0:
            if self.global_t > next_save_t * self.save_freq:
                wall_t = time.time() - start_time
                logger.info('Total time: {:.2f} seconds'.format(wall_t))
                Mstep_per_hr = int(self.global_t / (wall_t / 60 / 60)) / 1000000
                logger.info('Average: {:.2f} Msteps / hr'.format(Mstep_per_hr))
                logger.info('Estimated time left: {:.2f} hr'.format((self.train_max_steps - self.global_t)/1000000 / Mstep_per_hr))
                logger.info("num_policy_updates:" + str(num_policy_update))
                logger.info("num_agent_updates:" + str(num_agent_update))
                logger.info("replay memory size={}".format(self.replay_buffer.__len__()))
                if self.use_sil:
                    logger.info("num_sil_updates:" + str(num_sil_update))
                    logger.info("sil memory size={}".format(self.sil_buffer.__len__()))
                if self.use_rollout:
                    logger.info("num_rollout_steps:" + str(local_t))
                    logger.info("num_rollout_updates:" + str(num_rollout_update))
                    logger.info("num_rollout_performed:" + str(rollout_ctr))
                    logger.info("num_rollout_added:" + str(added_rollout_ctr))
                    logger.info("rollout_success_rate:" + str(added_rollout_ctr/rollout_ctr*100)+'%')
                    logger.info("refresh memory size={}".format(self.rollout_buffer.__len__()))
                    # if added_rollout_ctr == 0:
                    #     avg_return_diff = return_diff
                    # else:
                    #     avg_return_diff = return_diff / added_rollout_ctr
                    # logger.info("avg (new_return - old_return)={}".format(avg_return_diff))

                logger.info('Now saving data. Please wait')
                wall_t_fname = self.folder + '/' + 'wall_t.' + str(next_save_t * self.save_freq) #str(self.global_t)
                epsilon_fname = self.folder + '/epsilon'
                with open(wall_t_fname, 'w') as f:
                    f.write("wall_t:" + str(wall_t)+'\n')
                    f.write("num_policy_updates:" + str(num_policy_update)+'\n')
                    f.write("num_agent_updates:" + str(num_agent_update)+'\n')
                    if self.use_sil:
                        f.write("num_sil_updates:" + str(num_sil_update)+'\n')
                    if self.use_rollout:
                        f.write("num_rollout_updates:" + str(num_rollout_update)+'\n')
                        f.write("num_rollout_performed:" + str(rollout_ctr)+'\n')
                        f.write("num_rollout_added:" + str(added_rollout_ctr)+'\n')
                        f.write("rollout_success_rate:" + str(added_rollout_ctr/rollout_ctr*100)+'%'+'\n')
                        f.write("num_rollout_steps:" + str(local_t))
                with open(epsilon_fname, 'w') as f:
                    f.write(str(self.epsilon))

                self.net.save(next_save_t * self.save_freq) #self.global_t)

                # self.replay_buffer.save(name=self.name, folder=self.folder, resize=False)
                fn = self.folder + '/' + self.name.replace('-', '_') + '-dqn-rewards.pkl'
                pickle.dump(self.rewards, open(fn, 'wb'), pickle.HIGHEST_PROTOCOL)
                logger.info('Data saved!')
                next_save_t += 1

            # log information
            status = ""
            if self.global_t-1 < self.observe:
                status = "observe"
            elif self.global_t-1 < self.observe + self.explore:
                status = "explore"
            else:
                status = "train"

            if (self.global_t-1) % 10000 < 50: #== 0:
                if self.use_human_advice:
                    log_data = (
                        state, self.global_t-1, self.epsilon,
                        self.psi, use_advice, action, np.max(readout_t))
                    logger.debug(
                        "{0:}: global_t={1:} epsilon={2:.4f} psi={3:.4f} \
                        advice={4:} action={5:} q_max={6:.4f}".format(*log_data))
                else:
                    log_data = (
                        status, self.global_t-1, self.epsilon,
                        action, np.max(readout_t))
                    logger.debug(
                        "{0:}: global_t={1:} epsilon={2:.4f} action={3:} "
                        "q_max={4:.4f}".format(*log_data))

def playGame():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)) as sess:
        with tf.device('/gpu:'+os.environ["CUDA_VISIBLE_DEVICES"]):
            train(sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
