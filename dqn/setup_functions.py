import os
import logging
import numpy as np
import random

from common.util import egreedy, get_action_index, make_movie, load_memory, compute_returns
from common.game_state import get_wrapper_by_name
from termcolor import colored

logger = logging.getLogger("dqn")

def setup_folder(args, env_name):
    if not os.path.exists(args.save_to+"dqn/"):
        os.makedirs(args.save_to+"dqn/")

    if args.folder is not None:
        folder = args.save_to+'dqn/{}_{}'.format(args.gym_env.replace('-', '_'), args.folder)
    else:
        folder = args.save_to+'dqn/{}_{}'.format(args.gym_env.replace('-', '_'), args.optimizer.lower())
        end_str = ''

        if args.unclipped_reward:
            end_str += '_rawreward'
        elif args.log_scale_reward:
            end_str += '_logreward'

        if args.transformed_bellman:
            end_str += '_TB'
        if args.target_consistency:
            end_str += '_tcloss'

        if args.use_transfer:
            end_str += '_transfer'
            if args.not_transfer_conv2:
                end_str += '_noconv2'
            elif args.not_transfer_conv3 and args.use_mnih_2015:
                end_str += '_noconv3'
            elif args.not_transfer_fc1:
                end_str += '_nofc1'
            elif args.not_transfer_fc2:
                end_str += '_nofc2'

        if args.observe == 0:
            end_str += '_obs0'
        if args.init_epsilon < 1.0:
            end_str += '_lowinitexp'
        if args.use_human_model_as_advice:
            end_str += '_modelasadvice'

        if args.weight_decay is not None:
            end_str += '_wdecay'

        if args.priority_memory:
            end_str += '_prioritymem'
            if args.prio_by_return:
                end_str += '_byG'
            elif args.prio_by_td:
                end_str += '_byTD'

        if args.double_q:
            end_str += "_doubleq"

        if args.use_sil:
            end_str += '_sil'
        if args.sil_priority_memory:
            end_str += 'prio'
        if args.update_q_with_return:
            end_str += '_updtQbyG'

        if args.use_rollout:
            end_str += '_rollout'
            if args.update_in_rollout:
                end_str += "_updatein"

        if args.memory_length != 1000000:
            t="_"+str(args.memory_length / 1000000)+"memlen"
            end_str+=t

        folder += end_str

    if args.append_experiment_num is not None:
        folder += '_' + args.append_experiment_num

    return folder

def setup_demofolder(folder, env_name):
    if folder is not None:
        demo_memory_folder = folder
    else:
        demo_memory_folder = 'collected_demo/{}'.format(env_name)

    return demo_memory_folder

def setup_demo_memory(is_load_memory, use_sil, demo_ids, folder):
    demo_memory = None
    if is_load_memory and use_sil:
        demo_memory, _, _, _ = load_memory(name=None,
            demo_memory_folder=folder,
            demo_ids=demo_ids)

    return demo_memory

def setup_demo_memory_cam(is_load_demo_cam, demo_cam_id, folder):
    demo_memory_cam = None
    if is_load_demo_cam:
        demo_cam, _, _, _ = load_memory(name=None,
            demo_memory_folder=folder,
            demo_ids=demo_cam_id)

        demo_cam = demo_cam[int(demo_cam_id)]
        demo_memory_cam = np.zeros(
            (len(demo_cam),
             demo_cam.height,
             demo_cam.width,
             demo_cam.phi_length),
            dtype=np.float32)
        for i in range(len(demo_cam)):
            s0 = (demo_cam[i])[0]
            demo_memory_cam[i] = np.copy(s0)
        del demo_cam

    return demo_memory_cam

def setup_gpu(tf, cpu_only, gpu_fraction):
    gpu_options = None
    if not cpu_only:
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction)
    return gpu_options

def test(global_t, eval_game_state, eval_max_steps, net, n_actions, folder,
         rewards, render=False):
    logger.info("Evaluate policy at global_t={}...".format(global_t))

    episode_buffer = []
    eval_game_state.reset(hard_reset=True)
    episode_buffer.append(eval_game_state.get_screen_rgb())

    max_steps = eval_max_steps
    total_reward = 0
    total_steps = 0
    episode_reward = 0
    episode_steps = 0
    n_episodes = 0

    # # use one demonstration data to record cam
    # # only need to make movie for demo data once
    # # if self.global_t == 0:
    # cam, state, action = self.calculate_cam(self.test_cam_si)
    # cam_plus_img = []
    # cam_side_img = []
    #
    # for i in range(len(cam)):
    #     # overlay cam-state
    #     overlay = np.uint8(cam[i]).copy()
    #     output = np.uint8(state[i]).copy()
    #     alpha = 0.3
    #     cv2.addWeighted(overlay, alpha, output, 1 - alpha,
    #         0, output)
    #     # create a title space for action
    #     title_space = np.zeros((20, 84, 3), np.uint8)
    #     title_space[:] = (255,255,255)
    #     cv2.putText(title_space, "{}".format(ACTION_MEANING[action[i]]),
    #         (20, 14), cv2.FONT_HERSHEY_DUPLEX, .4, (0, 0, 0), 1)
    #     # concate title and state
    #     vcat_output = cv2.vconcat((title_space, output))
    #     cam_plus_img.append(vcat_output)
    #
    #     # side-by-side cam-state
    #     hcat_cam_state =  cv2.hconcat((np.uint8(cam[i]).copy(),
    #                                    np.uint8(state[i]).copy()))
    #     title_space = np.zeros((20, 84*2, 3), np.uint8)
    #     title_space[:] = (255,255,255)
    #     vcat_title_camstate = cv2.vconcat((title_space, hcat_cam_state))
    #     cv2.putText(vcat_title_camstate, "{}".format(ACTION_MEANING[action[i]]),
    #         (20, 14), cv2.FONT_HERSHEY_DUPLEX, .4, (0, 0, 0), 1)
    #     cam_side_img.append(vcat_title_camstate)
    #
    # time_per_step = 0.0167
    # make_movie(
    #     cam_plus_img,
    #     self.folder + '/frames/demo-cam_plus_img{ep:010d}'.format(ep=(self.global_t)),
    #     duration=len(cam)*time_per_step,
    #     true_image=True,
    #     salience=False)
    # make_movie(
    #     cam_side_img,
    #     self.folder + '/frames/demo-cam_side_img{ep:010d}'.format(ep=(self.global_t)),
    #     duration=len(state)*time_per_step,
    #     true_image=True,
    #     salience=False)
    # del cam, state, action, cam_plus_img, cam_side_img

    while max_steps > 0:
        readout_t = net.evaluate(eval_game_state.s_t)[0]
        action = get_action_index(readout_t,
                                  is_random=(random.random() <= 0.05),
                                  n_actions=n_actions)

        # take action
        eval_game_state.step(action)
        terminal = eval_game_state.terminal

        if n_episodes == 0 and global_t % 2000000 == 0:
            episode_buffer.append(eval_game_state.get_screen_rgb())

        episode_reward += eval_game_state.reward
        episode_steps += 1
        max_steps -= 1

        # s_t = s_t1
        eval_game_state.update()

        if terminal:
            if get_wrapper_by_name(eval_game_state.env, 'EpisodicLifeEnv').was_real_done:
                if n_episodes == 0 and global_t % 2000000 == 0:
                    time_per_step = 0.0167
                    images = np.array(episode_buffer)
                    make_movie(
                        images, folder + '/frames/image{ep:010d}'.format(ep=(global_t)),
                        duration=len(images)*time_per_step,
                        true_image=True, salience=False)
                    episode_buffer = []
                n_episodes += 1
                score_str = colored("score={}".format(episode_reward), "magenta")
                steps_str = colored("steps={}".format(episode_steps), "blue")
                log_data = (global_t, n_episodes, score_str, steps_str, total_steps)
                logger.debug("test: global_t={} trial={} {} {} total_steps={}".format(*log_data))
                total_reward += episode_reward
                total_steps += episode_steps
                episode_reward = 0
                episode_steps = 0
            eval_game_state.reset(hard_reset=False)

    if n_episodes == 0:
        total_reward = episode_reward
        total_steps = episode_steps
    else:
        # (timestep, total sum of rewards, total # of steps before terminating)
        total_reward = total_reward / n_episodes
        total_steps = total_steps // n_episodes

    log_data = (global_t, total_reward, total_steps, n_episodes)
    logger.debug("test: global_t={} final score={} final steps={} # episodes={}".format(*log_data))
    net.record_summary(
        score=total_reward, steps=total_steps,
        episodes=n_episodes, global_t=global_t, mode='Test')

    rewards['eval'][global_t] = (total_reward, total_steps, n_episodes)
    return total_reward, total_steps, n_episodes, rewards

def setup_pretrained_model(tf, args, action_size, in_shape, device=None):
    pretrain_model = None
    pretrain_graph = tf.Graph()
    if args.sae_classify_demo:
        # from class_network import AutoEncoderNetwork \
        #     as PretrainedModelNetwork
        logger.error("NotImplemented!")
        assert False
    elif args.classify_demo:
        from class_network import MultiClassNetwork \
            as PretrainedModelNetwork
    else:
        logger.error("Classification type Not supported yet!")
        assert False

    PretrainedModelNetwork.use_mnih_2015 = args.use_mnih_2015
    PretrainedModelNetwork.l1_beta = args.class_l1_beta
    PretrainedModelNetwork.l2_beta = args.class_l2_beta
    PretrainedModelNetwork.use_gpu = args.use_gpu
    # pretrained_model thread has to be -1!
    pretrain_model = PretrainedModelNetwork(
        pretrain_graph, action_size, -1,
        padding=args.padding,
        in_shape=in_shape, sae=args.sae_classify_demo,
        tied_weights=args.class_tied_weights,
        use_denoising=args.class_use_denoising,
        noise_factor=args.class_noise_factor,
        loss_function=args.class_loss_function,
        use_slv=args.class_use_slv, device=device)

    return pretrain_graph, pretrain_model

def setup_class_optimizer(tf, args):
    ae_opt = None
    opt = None
    if args.class_optimizer == 'rms':
        if args.ae_classify_demo:
            ae_opt = tf.train.RMSPropOptimizer(
                learning_rate=args.class_learn_rate,
                decay=args.class_opt_alpha,
                epsilon=args.class_opt_epsilon,
                )
        opt = tf.train.RMSPropOptimizer(
            learning_rate=args.class_learn_rate,
            decay=args.class_opt_alpha,
            epsilon=args.class_opt_epsilon,
            )

    else:  # Adam
        # Tensorflow defaults
        beta1 = 0.9
        beta2 = 0.999
        if args.ae_classify_demo:
            ae_opt = tf.train.AdamOptimizer(
                learning_rate=args.class_learn_rate,
                beta1=beta1, beta2=beta2,
                epsilon=args.class_opt_epsilon,
                )
        opt = tf.train.AdamOptimizer(
            learning_rate=args.class_learn_rate,
            beta1=beta1, beta2=beta2,
            epsilon=args.class_opt_epsilon,
            )
    return ae_opt, opt

def setup_common_worker(CommonWorker, args, action_size):
    CommonWorker.action_size = action_size
    CommonWorker.env_id = args.gym_env
    CommonWorker.reward_constant = args.reward_constant
    CommonWorker.max_global_time_step = args.max_time_step
    if args.unclipped_reward:
        CommonWorker.reward_type = "RAW"
    elif args.log_scale_reward:
        CommonWorker.reward_type = "LOG"
    else:
        CommonWorker.reward_type = "CLIP"
    # return CommonWorker

def setup_a3c_worker(A3CTrainingThread, args, log_idx):
    A3CTrainingThread.log_interval = args.log_interval
    A3CTrainingThread.perf_log_interval = args.performance_log_interval
    A3CTrainingThread.local_t_max = args.local_t_max
    A3CTrainingThread.entropy_beta = args.entropy_beta
    A3CTrainingThread.gamma = args.gamma
    A3CTrainingThread.use_mnih_2015 = args.use_mnih_2015
    A3CTrainingThread.finetune_upper_layers_only = args.finetune_upper_layers_only
    A3CTrainingThread.transformed_bellman = args.transformed_bellman
    A3CTrainingThread.clip_norm = args.grad_norm_clip
    A3CTrainingThread.use_grad_cam = args.use_grad_cam
    A3CTrainingThread.use_sil = args.use_sil
    A3CTrainingThread.use_sil_neg = args.use_sil_neg  # test if also using samples that are (G-V<0)
    A3CTrainingThread.log_idx = log_idx
    A3CTrainingThread.reward_constant = args.reward_constant

def setup_transfer_folder(args, env_name):
    if args.transfer_folder is not None:
            transfer_folder = args.transfer_folder
    else:
        transfer_folder = 'results/pretrain_models/{}'.format(env_name)
        end_str = ''
        if args.use_mnih_2015:
            end_str += '_mnih2015'
        end_str += '_l2beta1E-04_oversample'  # TODO(make this an argument)
        transfer_folder += end_str

    return transfer_folder

def setup_transfer_vars(args, network, folder):
    if args.not_transfer_conv2:
        transfer_var_list = [
            network.W_conv1,
            network.b_conv1,
            ]

    elif (args.not_transfer_conv3 and args.use_mnih_2015):
        transfer_var_list = [
            network.W_conv1,
            network.b_conv1,
            network.W_conv2,
            network.b_conv2,
            ]

    elif args.not_transfer_fc1:
        transfer_var_list = [
            network.W_conv1,
            network.b_conv1,
            network.W_conv2,
            network.b_conv2,
            ]

        if args.use_mnih_2015:
            transfer_var_list += [
                network.W_conv3,
                network.b_conv3,
                ]

    elif args.not_transfer_fc2:
        transfer_var_list = [
            network.W_conv1,
            network.b_conv1,
            network.W_conv2,
            network.b_conv2,
            network.W_fc1,
            network.b_fc1,
            ]

        if args.use_mnih_2015:
            transfer_var_list += [
                network.W_conv3,
                network.b_conv3,
                ]

    else:
        transfer_var_list = [
            network.W_conv1,
            network.b_conv1,
            network.W_conv2,
            network.b_conv2,
            network.W_fc1,
            network.b_fc1,
            network.W_fc2,
            network.b_fc2,
            ]

        if args.use_mnih_2015:
            transfer_var_list += [
                network.W_conv3,
                network.b_conv3,
                ]

        if '_slv' in str(folder):
            transfer_var_list += [
                network.W_fc3,
                network.b_fc3,
                ]

    return transfer_var_list

def initialize_uninitialized(tf, sess, model=None):
    if model is not None:
        with model.graph.as_default():
            global_vars=tf.global_variables()
    else:
        global_vars = tf.global_variables()

    is_not_initialized = sess.run(
        [tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = \
        [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))