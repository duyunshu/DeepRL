#!/usr/bin/env python3
import numpy as np
import logging
import os

# from common.replay_memory import ReplayMemory
from common.replay_memory.priority_memory import ReplayBuffer, PrioritizedReplayBuffer
from common.game_state import GameState
from common.util.schedules import LinearSchedule
from setup_functions import *
from rollout_thread import RolloutThread

logger = logging.getLogger("dqn")

def run_dqn(args):
    """
    Baseline:
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --cuda-devices=0 --optimizer=Adam --lr=0.0001 --decay=0.0 --momentum=0.0 --epsilon=0.001 --gpu-fraction=0.222
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --cuda-devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.00001 --gpu-fraction=0.222

    Transfer with Human Memory:
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --cuda-devices=0 --optimizer=Adam --lr=0.0001 --decay=0.0 --momentum=0.0 --epsilon=0.001 --observe=0 --use-transfer --load-memory
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --cuda-devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.00001 --observe=0 --use-transfer --load-memory
    python3 run_experiment.py breakout --cuda-devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.01 --observe=0 --use-transfer --load-memory --train-max-steps=20500000

    Transfer with Human Advice and Human Memory:
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --cuda-devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.00001 --observe=0 --use-transfer --load-memory --use-human-model-as-advice --advice-confidence=0. --psi=0.9999975 --train-max-steps=20500000

    Human Advice only with Human Memory:
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --cuda-devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.00001 --observe=0 --load-memory --use-human-model-as-advice --advice-confidence=0.75 --psi=0.9999975
    """
    from dqn_net import DqnNet
    from dqn_net_class import DqnNetClass
    from dqn_training import DQNTraining

    # setup folder
    GYM_ENV_NAME = args.gym_env.replace('-', '_')
    folder = setup_folder(args, GYM_ENV_NAME)

    # setup GPU options
    device = "/cpu:0"
    device_gpu = "/gpu:0"
    import tensorflow as tf
    gpu_options = setup_gpu(tf, args.cpu_only, args.gpu_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
        allow_soft_placement=True)

    game_state = GameState(env_id=args.gym_env, display=False, no_op_max=30,
                           human_demo=False, episode_life=True)
    n_actions = game_state.env.action_space.n

    eval_game_state = GameState(env_id=args.gym_env, display=False,
                                no_op_max=30, human_demo=False, episode_life=True)

    human_net = None
    sess_human = None
    if args.use_human_model_as_advice:
        if args.advice_folder is not None:
            advice_folder = args.advice_folder
        else:
            advice_folder = "{}_networks_classifier_{}".format(args.gym_env.replace('-', '_'), "adam")
        DqnNetClass.use_gpu = not args.cpu_only
        human_net = DqnNetClass(
            args.resized_height, args.resized_width,
            args.phi_len, game_state.env.action_space.n, args.gym_env,
            optimizer="Adam", learning_rate=0.0001, epsilon=0.001,
            decay=0., momentum=0., folder=advice_folder,
            device=device if args.cpu_only else device_gpu)
        sess_human = tf.Session(config=config, graph=human_net.graph)
        human_net.initializer(sess_human)
        human_net.load()

    # prepare session
    sess = tf.Session(config=config)

    replay_buffer = None
    prioritized_replay_beta_iters = None
    prioritized_replay_eps = None
    beta_schedule = None
    if args.priority_memory:
        replay_buffer = PrioritizedReplayBuffer(size=args.memory_length,
                                                alpha=args.prio_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = 50*10**6 #args.train_max_steps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=args.prio_init_beta,
                                       final_p=1.0)
        prioritized_replay_eps = 1e-6 #TODO: add to parameter list
    else:
        replay_buffer = ReplayBuffer(size=args.memory_length)
        beta_schedule=None
        prioritized_replay_eps=0


    sil_buffer = None
    if args.use_sil:
        if args.sil_priority_memory:
            sil_buffer = PrioritizedReplayBuffer(size=args.memory_length,
                                                 alpha=0.6)
            assert prioritized_replay_eps is not None
        else:
            sil_buffer = ReplayBuffer(size=args.memory_length)
            beta_schedule=None
            prioritized_replay_eps=0


    rollout_buffer = None
    temp_buffer = None
    if args.use_rollout:
        if args.priority_memory:
            rollout_buffer = PrioritizedReplayBuffer(size=args.memory_length,
                                                     alpha=0.6)
            temp_buffer = PrioritizedReplayBuffer(size=2*args.batch,
                                                  alpha=0.6)
            # these should have been set in priori_mem already
            assert prioritized_replay_beta_iters is not None
            assert prioritized_replay_eps is not None
            assert beta_schedule is not None

        else:
            rollout_buffer = ReplayBuffer(size=args.memory_length)
            temp_buffer = ReplayBuffer(size=2*args.batch)
            beta_schedule=None
            prioritized_replay_eps=0

    # baseline learning
    if not args.use_transfer:
        # DqnNet.use_gpu = not args.cpu_only
        net = DqnNet(
            sess, args.resized_height, args.resized_width, args.phi_len,
            n_actions, args.gym_env, gamma=args.gamma,
            optimizer=args.optimizer, learning_rate=args.lr,
            epsilon=args.epsilon, decay=args.decay, momentum=args.momentum,
            verbose=args.verbose, folder=folder,
            slow=args.use_slow, tau=args.tau,
            device=device if args.cpu_only else device_gpu,
            transformed_bellman=args.transformed_bellman,
            target_consistency_loss=args.target_consistency,
            clip_norm=args.grad_norm_clip,
            weight_decay=args.weight_decay,
            use_rollout=args.use_rollout, use_sil=args.use_sil,
            double_q=args.double_q)
    # transfer using existing model
    else:
        if args.transfer_folder is not None:
            transfer_folder = args.transfer_folder
        else:
            transfer_folder = 'results/pretrain_models/{}'.format(args.gym_env.replace('-', '_'))
            end_str = ''
            end_str += '_mnih2015'
            end_str += '_l2beta1E-04_batchprop'  #TODO: make this an argument
            transfer_folder += end_str

        transfer_folder += '/transfer_model'

        DqnNet.use_gpu = not args.cpu_only
        net = DqnNet(
            sess, args.resized_height, args.resized_width, args.phi_len,
            n_actions, args.gym_env, gamma=args.gamma,
            optimizer=args.optimizer, learning_rate=args.lr,
            epsilon=args.epsilon, decay=args.decay, momentum=args.momentum,
            verbose=args.verbose, folder=folder,
            slow=args.use_slow, tau=args.tau,
            transfer=True, transfer_folder=transfer_folder,
            not_transfer_conv2=args.not_transfer_conv2,
            not_transfer_conv3=args.not_transfer_conv3,
            not_transfer_fc1=args.not_transfer_fc1,
            not_transfer_fc2=args.not_transfer_fc2,
            device=device if args.cpu_only else device_gpu,
            transformed_bellman=args.transformed_bellman,
            target_consistency_loss=args.target_consistency,
            clip_norm=args.grad_norm_clip,
            weight_decay=args.weight_decay,
            is_rollout_net=False)

    if args.unclipped_reward:
        reward_type = 'RAW'
    elif args.log_scale_reward:
        reward_type = 'LOG'
    else:
        reward_type = 'CLIP'

    # rollout thread
    rollout_worker = None
    if args.use_rollout:
        rollout_worker = RolloutThread(
                    action_size=n_actions, env_id=args.gym_env,
                    update_in_rollout=args.update_in_rollout,
                    global_pretrained_model=None, #TODO
                    local_pretrained_model=None, #TODO
                    transformed_bellman = args.transformed_bellman,
                    no_op_max=0,
                    device=device if args.cpu_only else device_gpu,
                    clip_norm=args.grad_norm_clip, nstep_bc=0,
                    reward_type=reward_type)

    ##added load human demonstration for testing cam
    demo_memory_folder = None
    demo_ids = None
    if args.load_memory or args.load_demo_cam:
        if args.demo_memory_folder is not None:
            demo_memory_folder = args.demo_memory_folder
        else:
            demo_memory_folder = 'collected_demo/{}'.format(args.gym_env.replace('-', '_'))
        # demo_ids = tuple(map(int, args.demo_ids.split(",")))

    experiment = DQNTraining(
        sess, net, game_state, eval_game_state, args.resized_height, args.resized_width,
        args.phi_len, n_actions, args.batch, args.gym_env,
        args.gamma, args.observe, args.explore, args.final_epsilon,
        args.init_epsilon, replay_buffer,
        args.update_freq, args.save_freq, args.eval_freq,
        args.eval_max_steps, args.c_freq,
        folder, load_demo_memory=args.load_memory, demo_ids=args.demo_ids,
        load_demo_cam=args.load_demo_cam, demo_cam_id=args.demo_cam_id,
        demo_memory_folder=demo_memory_folder,
        train_max_steps=args.train_max_steps,
        human_net=human_net, confidence=args.advice_confidence, psi=args.psi,
        train_with_demo_steps=args.train_with_demo_steps,
        use_transfer=args.use_transfer, reward_type=reward_type,
        priority_memory=args.priority_memory, beta_schedule=beta_schedule,
        prioritized_replay_eps=prioritized_replay_eps,
        prio_by_td=args.prio_by_td, prio_by_return=args.prio_by_return,
        update_q_with_return=args.update_q_with_return,
        use_rollout=args.use_rollout, rollout_buffer=rollout_buffer,
        rollout_worker=rollout_worker, temp_buffer=temp_buffer,
        update_in_rollout=args.update_in_rollout,
        use_sil=args.use_sil, sil_buffer=sil_buffer,
        sil_priority_memory=args.sil_priority_memory)

    experiment.run()

    if args.use_human_model_as_advice:
        sess_human.close()

    sess.close()
