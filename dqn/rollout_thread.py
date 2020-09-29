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
# from common.replay_memory import ReplayMemory
from common.util import generate_image_for_cam_video
from common.util import egreedy, get_action_index, make_movie, load_memory, compute_returns
from common.util import grad_cam
from common.util import make_movie
from common.util import transform_h
from common.util import transform_h_inv
from common.util import visualize_cam
from termcolor import colored
from queue import Queue
from copy import deepcopy
from setup_functions import test

logger = logging.getLogger("rollout_thread")


class RolloutThread(object):
    """Rollout Thread Class."""
    advice_confidence = 0.8
    gamma = 0.99

    def __init__(self, action_size, env_id,
                 update_in_rollout, #rollout_buffer,
                 global_pretrained_model=None, local_pretrained_model=None,
                 transformed_bellman=False, no_op_max=0,
                 device='/cpu:0', clip_norm=None, nstep_bc=0, reward_type='CLIP'):
        """Initialize RolloutThread class."""
        self.is_rollout_thread = True
        self.action_size = action_size
        self.transformed_bellman = transformed_bellman
        self.clip_norm = clip_norm
        # self.learning_rate_input = learning_rate_input
        self.reward_constant = 2 # TODO: add to parameters
        self.reward_type = reward_type

        self.no_op_max = no_op_max
        self.override_num_noops = 0 if self.no_op_max == 0 else None

        logger.info("===ROLLOUT thread===")
        logger.info("action_size: {}".format(self.action_size))
        logger.info("reward_type: {}".format(self.reward_type))
        logger.info("transformed_bellman: {}".format(
            colored(self.transformed_bellman,
                    "green" if self.transformed_bellman else "red")))
        logger.info("update in rollout: {}".format(
            colored(update_in_rollout, "green" if update_in_rollout else "red")))

        self.reward_clipped = True if self.reward_type == 'CLIP' else False

        # # setup local dqn (self-rollout)
        # self.local_dqn = local_net
        # self.sync_dqn = self.local_dqn.sync_from(global_net)
        # with tf.device(device):
        #     local_vars = self.local_a3c.get_vars
        #     self.local_a3c.prepare_loss(
        #         entropy_beta=self.entropy_beta, critic_lr=0.5)
        #     var_refs = [v._ref() for v in local_vars()]
        #     self.gradients = tf.gradients(self.local_a3c.total_loss, var_refs)
        #     global_vars = global_a3c.get_vars
        #     if self.clip_norm is not None:
        #         self.gradients, grad_norm = tf.clip_by_global_norm(
        #             self.gradients, self.clip_norm)
        #     self.gradients = list(zip(self.gradients, global_vars()))
        #     self.apply_gradients = grad_applier.apply_gradients(self.gradients)

        # TODO: setup local pretrained model
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

        # assert self.local_dqn is not None
        if nstep_bc > 0:
            assert self.local_pretrained is not None

        # self.episode = SILReplayMemory(
        #     self.action_size, max_len=None, gamma=self.gamma,
        #     clip=self.reward_clipped,
        #     height=self.local_a3c.in_shape[0],
        #     width=self.local_a3c.in_shape[1],
        #     phi_length=self.local_a3c.in_shape[2],
        #     reward_constant=self.reward_constant)


    def record_rollout(self, score=0, steps=0,
                       old_return=0, new_return=0,
                       global_t=0, rollout_ctr=0, added_rollout_ctr=0,
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
        summary.value.add(tag='{}/added_rollout_ctr'.format(mode),
                          simple_value=float(added_rollout_ctr))
        if confidence is not None:
            summary.value.add(tag='{}/advice-confidence'.format(mode),
                              simple_value=float(confidence))
        if episodes is not None:
            summary.value.add(tag='{}/episodes'.format(mode),
                              simple_value=float(episodes))
        self.writer.add_summary(summary, global_t)
        self.writer.flush()

    # def compute_return_for_state(self, rewards, terminal):
    #     """Compute expected return for a given state."""
    #     assert self.reward_clipped, "TB not supported yet!" #TODO
    #     length = np.shape(rewards)[0]
    #     returns = np.empty_like(rewards, dtype=np.float32)
    #
    #     if self.reward_clipped:
    #         rewards = np.clip(rewards, -1., 1.)
    #     else:
    #         rewards = np.sign(rewards) * self.reward_constant + rewards
    #
    #     for i in reversed(range(length)):
    #         if terminal[i]:
    #             returns[i] = rewards[i] if self.reward_clipped else transform_h(rewards[i])
    #         else:
    #             if self.reward_clipped:
    #                 returns[i] = rewards[i] + self.gamma * returns[i+1]
    #             else:
    #                 # apply transformed expected return
    #                 exp_r_t = self.gamma * transform_h_inv(returns[i+1])
    #                 returns[i] = transform_h(rewards[i] + exp_r_t)
    #     return returns[0]

    def update_dqn(self, sess, net, actions, states, rewards, values, global_t):
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

        sess.run(self.apply_gradients, feed_dict=feed_dict)

    def rollout(self, dqn_sess, global_net, global_t,
                samplestate, rollout_ctr, added_rollout_ctr, add_all_rollout,
                eval_freq, reward_dict, eval_game_state,
                eval_max_steps, n_actions, folder,
                ep_max_steps=10000, nstep_bc=0, update_in_rollout=False):
        """Rollout, one at a time."""
        # dqn_sess.run(self.sync_dqn)
        assert nstep_bc == 0, "BC not supported yet!" #TODO
        # if nstep_bc > 0:
        #     pretrain_sess.run(self.sync_pretrained)

        # assert pretrain_sess.run(self.local_pretrained.W_fc2).all() == \
        #     pretrain_sess.run(global_pretrained.W_fc2).all()

        # for each bad state in queue, do rollout till terminal (no max=20 limit)
        # obses_t, fss, actions, rewards, obses_t_next, dones, returns, from_rollout
        _, fs, old_a, _, _, _, old_return, _ = samplestate
        fs = fs.flatten()

        states = []
        fullstates = []
        actions = []
        rewards = []
        terminals = []
        next_states = []

        terminal_pseudo = False  # loss of life
        terminal_end = False  # real terminal
        add = False

        self.rolloutgame.reset(hard_reset=True)
        self.rolloutgame.restore_full_state(fs) # only works with 1-D array!
        # check if restore successful
        # assert self.rolloutgame.s_t.all() == fs.all()
        fs_check = self.rolloutgame.clone_full_state()
        assert fs_check.all() == fs.all()
        del fs_check

        start_local_t = self.local_t
        n_actions= self.rolloutgame.env.action_space.n
        # self.rolloutgame.step(0)
        # self.rolloutgame.update()

        # # record video of rollout
        # video_buffer = []
        # init_img = self.rolloutgame.get_screen_rgb()
        # video_buffer.append(init_img)

        # prevent breakout from stucking,
        # see https://github.com/openai/gym/blob/54f22cf4db2e43063093a1b15d968a57a32b6e90/gym/envs/__init__.py#L635
        while ep_max_steps > 0 and not add: #True:
            # print(ep_max_steps)
            # self.rolloutgame.env.render()
            # time.sleep(0.01)

            if global_t % eval_freq == 0:
                total_reward, total_steps, n_episodes, reward_dict = \
                        test(global_t, eval_game_state,
                             eval_max_steps, global_net, n_actions,
                             folder, reward_dict)

            state = self.rolloutgame.s_t
            fullstate = self.rolloutgame.clone_full_state()

            # if nstep_bc > 0:
            #     # print("taking action from BC, {} steps left".format(nstep_bc))
            #     model_pi = self.local_pretrained.run_policy(pretrain_sess, state)
            #     if game == "Breakout": # breakout needs some stocasity
            #         action = self.egreedy_action(model_pi, epsilon=0.01)
            #         confidences.append(model_pi[action])
            #     else:
            #         action, confidence = self.choose_action_with_high_confidence(
            #                                   model_pi, exclude_noop=False)
            #         confidences.append(confidence)
            #     nstep_bc -= 1
            # else:
            readout_t = global_net.evaluate(self.rolloutgame.s_t)[0]
            action = get_action_index(readout_t, is_random=False,
                                      n_actions=n_actions)

            states.append(state)
            fullstates.append(fullstate)
            actions.append(action)

            next_state, reward, terminal = self.rolloutgame.step(action)
            terminals.append(terminal)
            next_states.append(next_state)

            ep_max_steps-=1

            # video_buffer.append(self.rolloutgame.get_screen_rgb())

            self.episode_reward += reward

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
                ep_max_steps = 0
                terminal_pseudo = True
                env = self.rolloutgame.env
                name = 'EpisodicLifeEnv'
                rollout_ctr += 1
                terminal_end = get_wrapper_by_name(env, name).was_real_done
                if rollout_ctr % 100 == 0 and rollout_ctr > 0:
                    log_msg = "ROLLOUT: rollout_ctr={} added_rollout_ct={} global_t={} local_t={}".format(
                        rollout_ctr, added_rollout_ctr, global_t, self.local_t)
                    score_str = colored("score={}".format(
                        self.episode_reward), "magenta")
                    steps_str = colored("steps={}".format(
                        self.episode_steps), "blue")
                    # conf_str = colored("advice-confidence={}".format(
                    #     np.mean(confidences)), "blue")
                    log_msg += " {} {}".format(score_str, steps_str)#, conf_str)
                    logger.info(log_msg)

                new_return = compute_returns(rewards, terminals)
                # print("new&old returns: ",new_return, old_return)
                if not add_all_rollout:
                    if new_return[0] > old_return:
                        add = True
                        added_rollout_ctr += 1
                else:
                    add = True
                    added_rollout_ctr += 1

                self.episode_reward = 0
                self.episode_steps = 0
                self.rolloutgame.reset(hard_reset=True)
                break

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

                # round_t = int(math.floor(global_t/10))*10
                # rollout_dict["added_ctr"][round_t] = added_rollout_ctr
                # rollout_dict["total_ctr"][round_t] = rollout_ctr
                # rollout_dict["new_return"][round_t] = new_return[0]
                # rollout_dict["old_return"][round_t] = old_return

                # self.record_rollout(
                #     score=self.episode_reward, steps=self.episode_steps,
                #     old_return=old_return, new_return=new_return[0],
                #     global_t=global_t, rollout_ctr=rollout_ctr,
                #     added_rollout_ctr=added_rollout_ctr,
                #     mode='Rollout',
                #     confidence=np.mean(confidences), episodes=None)

        diff_local_t = self.local_t - start_local_t

        if add:
            data = (states, fullstates, actions, rewards, next_states, terminals, new_return)
        else:
            data = None

        return self.local_t, diff_local_t, rollout_ctr, added_rollout_ctr, data, reward_dict

    def mixed_sampling(self, priority_mem, buffer, batch_size, beta_schedule, global_t):
        if priority_mem: # priority mem
            exp = buffer.sample(batch_size, beta=beta_schedule.value(global_t))
            s_j_batch, fs, a_batch, r_batch, s_j1_batch, t_batch, ret, roll, \
                weights, idxes = exp
            data = (s_j_batch, fs, a_batch, r_batch, s_j1_batch, t_batch, ret, roll)
        else: # non priority mem
            exp = buffer.sample(batch_size)
            s_j_batch, fs, a_batch, r_batch, s_j1_batch, t_batch, ret, roll = exp
            data = (s_j_batch, fs, a_batch, r_batch, s_j1_batch, t_batch, ret, roll)
            weights, idxes = np.ones_like(r_batch), None

        return data, weights, idxes

    def compute_priority(self, net, data, weights, global_t):
        s_j_batch, _, a_batch, r_batch, s_j1_batch, t_batch, _, _ = data
        actions_onehot = []
        for ai in a_batch:
            a = np.zeros([self.action_size])
            a[ai] = 1
            actions_onehot.append(a)
        td_errors = net.get_td_error(s_j_batch, actions_onehot, r_batch,
                                     s_j1_batch, t_batch, weights, global_t)
        return td_errors
