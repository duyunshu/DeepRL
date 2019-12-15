#!/usr/bin/env python3
"""Asynchronous Advantage Actor-Critic (A3C).

Usage:
python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --parallel-size=16
    --initial-learn-rate=7e-4 --use-lstm --use-mnih-2015

python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --parallel-size=16
    --initial-learn-rate=7e-4 --use-lstm --use-mnih-2015 --use-transfer
    --not-transfer-fc2 --transfer-folder=<>

python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --parallel-size=16
    --initial-learn-rate=7e-4 --use-lstm --use-mnih-2015 --use-transfer
    --not-transfer-fc2 --transfer-folder=<> --load-pretrained-model
    --onevsall-mtl --pretrained-model-folder=<>
    --use-pretrained-model-as-advice --use-pretrained-model-as-reward-shaping
"""
import logging
import numpy as np
import os
import pathlib
import signal
import sys
import threading
import time

from a3c_training_thread import A3CTrainingThread
from common.game_state import GameState
from common.util import load_memory
from common.util import prepare_dir
from game_ac_network import GameACFFNetwork
from game_ac_network import GameACLSTMNetwork
from queue import Queue
from sil_memory import SILReplayMemory
# from common.replay_memory import ReplayMemoryReturns
from copy import deepcopy

logger = logging.getLogger("a3c")

try:
    import cPickle as pickle
except ImportError:
    import pickle

def pause():
    int(input("enter a number to cont..."))


def run_a3c(args):
    """Run A3C experiment."""
    GYM_ENV_NAME = args.gym_env.replace('-', '_')

    if args.use_gpu:
        assert args.cuda_devices != ''
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import tensorflow as tf

    if not os.path.exists('scratch/cluster/yunshu/results/a3c'):
        os.makedirs('scratch/cluster/yunshu/results/a3c')

    if args.folder is not None:
        folder = 'scratch/cluster/yunshu/results/a3c/{}_{}'.format(GYM_ENV_NAME, args.folder)
    else:
        folder = 'scratch/cluster/yunshu/results/a3c/{}'.format(GYM_ENV_NAME)
        end_str = ''

        if args.use_mnih_2015:
            end_str += '_mnih2015'
        if args.padding == 'SAME':
            end_str += '_same'
        if args.use_lstm:
            end_str += '_lstm'
        if args.unclipped_reward:
            end_str += '_rawreward'
        elif args.log_scale_reward:
            end_str += '_logreward'
        if args.transformed_bellman:
            end_str += '_transformedbell'

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
        if args.finetune_upper_layers_only:
            end_str += '_tune_upperlayers'

        if args.load_pretrained_model:
            if args.use_pretrained_model_as_advice:
                end_str += '_modelasadvice'
            if args.use_pretrained_model_as_reward_shaping:
                end_str += '_modelasshaping'
            if args.use_correction:
                end_str+='_usecorrection'

        if args.use_sil:
            end_str += '_sil'
            if args.priority_memory:
                end_str += '_prioritymem'
            if args.use_sil_neg:
                end_str+= '_usenegsample'

        folder += end_str

    if args.append_experiment_num is not None:
        folder += '_' + args.append_experiment_num

    folder = pathlib.Path(folder)

    demo_memory = None
    if args.load_memory or args.load_demo_cam:
        if args.demo_memory_folder is not None:
            demo_memory_folder = args.demo_memory_folder
        else:
            demo_memory_folder = 'collected_demo/{}'.format(GYM_ENV_NAME)
        demo_memory_folder = pathlib.Path(demo_memory_folder)

    if args.load_memory and args.use_sil:
        demo_memory, actions_ctr, total_rewards, total_steps = \
            load_memory(
                name=None,
                demo_memory_folder=demo_memory_folder,
                demo_ids=args.demo_ids,
                )

    demo_memory_cam = None
    if args.load_demo_cam:
        demo_cam, _, total_rewards_cam, _ = load_memory(
            name=None,
            demo_memory_folder=demo_memory_folder,
            demo_ids=args.demo_cam_id,
            )

        demo_cam = demo_cam[int(args.demo_cam_id)]
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
        logger.info("loaded demo {} for testing CAM".format(args.demo_cam_id))

    device = "/cpu:0"
    device_gpu = ""
    gpu_options = None
    if args.use_gpu:
        device_gpu = "/gpu:"+os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=args.gpu_fraction)

    initial_learning_rate = args.initial_learn_rate
    logger.info('A3C Initial Learning Rate={}'.format(initial_learning_rate))
    class_initial_learning_rate = 0.
    if args.use_correction:
        class_initial_learning_rate = args.class_learn_rate
        logger.info('Classifier Initial Learning Rate={}'.format(class_initial_learning_rate))
    time.sleep(2)

    global_t = 0
    pretrain_global_t = 0
    pretrain_epoch = 0
    rewards = {'train': {}, 'eval': {}}
    best_model_reward = -(sys.maxsize)

    class_best_model_reward = -(sys.maxsize)
    class_rewards = {'class_eval': {}}

    stop_req = False

    game_state = GameState(env_id=args.gym_env)
    action_size = game_state.env.action_space.n
    game_state.close()
    del game_state.env
    del game_state

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
        allow_soft_placement=True)

    input_shape = (args.input_shape, args.input_shape, 4)

    global_pretrained_model = None
    local_pretrained_model = None
    pretrain_graph = None
    if args.load_pretrained_model:
        pretrain_graph = tf.Graph()
        if args.onevsall_mtl:
            logger.error("Not supported yet!")
            assert False
            # from game_class_network import MTLBinaryClassNetwork \
            #     as PretrainedModelNetwork
        else:
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

        if args.pretrained_model_folder is not None:
            pretrained_model_folder = args.pretrained_model_folder
        else:
            logger.error("please enter pretrained model folder!")
            assert False
            # pretrained_model_folder = GYM_ENV_NAME
            # pretrained_model_folder += '_classifier_use_mnih_onevsall_mtl'

        # pretrained_model_folder = pathlib.Path(pretrained_model_folder)

        PretrainedModelNetwork.use_mnih_2015 = args.use_mnih_2015
        PretrainedModelNetwork.l1_beta = args.class_l1_beta
        PretrainedModelNetwork.l2_beta = args.class_l2_beta
        PretrainedModelNetwork.use_gpu = args.use_gpu
        # pretrained_model thread has to be -1!
        global_pretrained_model = PretrainedModelNetwork(
            pretrain_graph, action_size, -1,
            device=device_gpu if args.use_gpu else device, padding=args.padding,
            in_shape=input_shape, sae=args.sae_classify_demo,
            tied_weights=args.class_tied_weights,
            use_denoising=args.class_use_denoising,
            noise_factor=args.class_noise_factor,
            loss_function=args.class_loss_function,
            use_slv=args.class_use_slv)

        logger.info("Classifier optimizer: {}".format(
            'RMSPropOptimizer' if args.class_optimizer == 'rms' \
                               else 'AdamOptimizer'))
        logger.info("Classifier learning_rate: {}".format(args.class_learn_rate))
        logger.info("Classifier epsilon: {}".format(args.class_opt_epsilon))
        if args.class_optimizer == 'rms':
            logger.info("\trms decay: {}".format(args.rmsp_alpha))
        else:  # Adam
            # Tensorflow defaults
            beta1 = 0.9
            beta2 = 0.999
        # class optimizer
        with tf.device(device_gpu if args.use_gpu else device):
            class_ae_opt = None
            if args.class_optimizer == 'rms':
                if args.ae_classify_demo:
                    class_ae_opt = tf.train.RMSPropOptimizer(
                        learning_rate=args.class_learn_rate,
                        decay=args.class_opt_alpha,
                        epsilon=args.class_opt_epsilon,
                        )
                class_opt = tf.train.RMSPropOptimizer(
                    learning_rate=args.class_learn_rate,
                    decay=args.class_opt_alpha,
                    epsilon=args.class_opt_epsilon,
                    )

            else:  # Adam
                if args.ae_classify_demo:
                    class_ae_opt = tf.train.AdamOptimizer(
                        learning_rate=args.class_learn_rate,
                        beta1=beta1, beta2=beta2,
                        epsilon=args.class_opt_epsilon,
                        )
                class_opt = tf.train.AdamOptimizer(
                    learning_rate=args.class_learn_rate,
                    beta1=beta1, beta2=beta2,
                    epsilon=args.class_opt_epsilon,
                    )

    if args.use_lstm:
        GameACLSTMNetwork.use_mnih_2015 = args.use_mnih_2015
        global_network = GameACLSTMNetwork(
            action_size, -1, device, padding=args.padding,
            in_shape=input_shape)
    else:
        GameACFFNetwork.use_mnih_2015 = args.use_mnih_2015
        global_network = GameACFFNetwork(
            action_size, -1, device, padding=args.padding,
            in_shape=input_shape)

    all_workers = []

    # a3c optimizer
    learning_rate_input = tf.placeholder(tf.float32, shape=(), name="opt_lr")
    grad_applier = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate_input,
        decay=args.rmsp_alpha,
        epsilon=args.rmsp_epsilon)


    A3CTrainingThread.log_interval = args.log_interval
    A3CTrainingThread.perf_log_interval = args.performance_log_interval
    A3CTrainingThread.local_t_max = args.local_t_max
    A3CTrainingThread.use_lstm = args.use_lstm
    A3CTrainingThread.action_size = action_size
    A3CTrainingThread.entropy_beta = args.entropy_beta
    A3CTrainingThread.gamma = args.gamma
    A3CTrainingThread.use_mnih_2015 = args.use_mnih_2015
    A3CTrainingThread.env_id = args.gym_env
    A3CTrainingThread.finetune_upper_layers_only = \
        args.finetune_upper_layers_only
    A3CTrainingThread.transformed_bellman = args.transformed_bellman
    A3CTrainingThread.clip_norm = args.grad_norm_clip
    A3CTrainingThread.use_grad_cam = args.use_grad_cam
    A3CTrainingThread.use_sil = args.use_sil
    A3CTrainingThread.use_sil_neg = args.use_sil_neg  # test if also using samples that are (G-V<0)
    A3CTrainingThread.use_correction = args.use_correction
    if args.use_sil:
        A3CTrainingThread.log_idx = 1
    if args.use_correction:
        A3CTrainingThread.log_idx = 3
        assert global_pretrained_model is not None
    A3CTrainingThread.reward_constant = args.reward_constant

    if args.unclipped_reward:
        A3CTrainingThread.reward_type = "RAW"
    elif args.log_scale_reward:
        A3CTrainingThread.reward_type = "LOG"
    else:
        A3CTrainingThread.reward_type = "CLIP"

    shared_memory = None
    exp_buffer = None
    if args.use_sil:
        shared_memory = SILReplayMemory(
            action_size, max_len=args.memory_length, gamma=args.gamma,
            clip=False if args.unclipped_reward else True,
            height=input_shape[0], width=input_shape[1],
            phi_length=input_shape[2], priority=args.priority_memory, #SIL uses priority
            reward_constant=args.reward_constant)

        if args.use_correction:
            exp_buffer = SILReplayMemory(
                action_size, max_len=args.memory_length, gamma=args.gamma,
                clip=False if args.unclipped_reward else True,
                height=input_shape[0], width=input_shape[1],
                phi_length=input_shape[2], priority=False, #classifier doesn't use priority
                reward_constant=args.reward_constant)

        if demo_memory is not None:
            temp_memory = SILReplayMemory(
                action_size, max_len=args.memory_length, gamma=args.gamma,
                clip=False if args.unclipped_reward else True,
                height=input_shape[0], width=input_shape[1],
                phi_length=input_shape[2], priority=False,
                reward_constant=args.reward_constant)
            if args.use_correction:
                temp_memory2 = SILReplayMemory(
                    action_size, max_len=args.memory_length, gamma=args.gamma,
                    clip=False if args.unclipped_reward else True,
                    height=input_shape[0], width=input_shape[1],
                    phi_length=input_shape[2], priority=False,
                    reward_constant=args.reward_constant)

            for idx in list(demo_memory.keys()):
                demo = demo_memory[idx]
                for i in range(len(demo)+1):
                    s0, a0, _, fs0, _, r1, t1, _ = demo[i]
                    temp_memory.add_item(s0, fs0, a0, r1, t1)
                    if args.use_correction:
                        temp_memory2.add_item(s0, fs0, a0, r1, t1)

                    if t1:  # terminal
                        shared_memory.fs_size = len(fs0)
                        shared_memory.extend(temp_memory)
                        if args.use_correction:
                            exp_buffer.fs_size = len(fs0)
                            exp_buffer.extend(temp_memory2)

                if len(temp_memory) > 0:
                    logger.warning("Disregard {} states in"
                                   " demo_memory {}".format(
                                    len(temp_memory), idx))
                    temp_memory.reset()
                if args.use_correction:
                    if len(temp_memory2) > 0:
                        logger.warning("Disregard {} states in"
                                       " demo_memory {}".format(
                                        len(temp_memory2), idx))
                        temp_memory2.reset()

            # log memory information
            shared_memory.log()
            del temp_memory

            if args.use_correction:
                exp_buffer.log()
                del temp_memory2

    # one sil thread, one classifier thread, one rollout thread, others are a3c
    if args.use_sil and args.use_correction:
        assert args.parallel_size >= 4

    n_shapers = args.parallel_size  # int(args.parallel_size * .25)
    mod = args.parallel_size // n_shapers
    for i in range(args.parallel_size):
        is_reward_shape = False
        is_advice = False
        sil_thread = i == 0 and args.use_sil
        classify_thread = i == 1 and args.use_correction
        rollout_thread = i == 2 and args.use_correction
        if i % mod == 0:
            is_reward_shape = args.use_pretrained_model_as_reward_shaping
            is_advice = args.use_pretrained_model_as_advice

        # classifier uses GPU
        if args.use_gpu:
            _device = device_gpu if classify_thread else device
        else:
            _device="/cpu:0"

        if args.use_lstm:
            # local_network = GameACLSTMNetwork(
            #     action_size, i, device, padding=args.padding,
            #     in_shape=input_shape)
            logger.error("Not supported yet!")
            assert False
        else:
            if not classify_thread and not rollout_thread:
                local_network = GameACFFNetwork(
                    action_size, i, device=_device,
                    padding=args.padding,
                    in_shape=input_shape)

                a3c_worker = A3CTrainingThread(
                    i, global_network, local_network, initial_learning_rate,
                    learning_rate_input,
                    grad_applier, args.max_time_step,
                    device=_device,
                    pretrained_model=None, #global_pretrained_model if sil_thread else None,
                    pretrained_model_sess=None,# global_pretrained_model_sess if sil_thread else None,
                    advice=is_advice,
                    correction=args.use_correction,
                    reward_shaping=is_reward_shape,
                    sil_thread=sil_thread,
                    classify_thread=classify_thread,
                    rollout_thread=rollout_thread,
                    batch_size=args.batch_size if sil_thread else None,
                    no_op_max=30)

            if classify_thread:
                if args.sae_classify_demo:
                    # from sae_classifier import AutoencoderClassifyDemo \
                    #     as ClassifierThread
                    logger.error("NotImplemented!")
                    assert False
                elif args.classify_demo:
                    from classifier_thread import ClassifyDemo \
                        as ClassifierThread
                else:
                    logger.error("Classifier Type Not supported yet!")
                    assert False

                assert global_pretrained_model is not None

                a3c_worker = ClassifierThread(
                    tf=tf, net=global_pretrained_model,
                    thread_index=i,
                    game_name=args.gym_env,
                    train_max_steps=int(args.class_train_steps),
                    batch_size=args.batch_size,
                    ae_grad_applier=class_ae_opt,
                    grad_applier=class_opt,
                    eval_freq=args.eval_freq,
                    device=_device, clip_norm=args.grad_norm_clip,
                    game_state=GameState(env_id=args.gym_env, no_op_max=30),
                    sampling_type=args.class_sample_type,
                    sl_loss_weight=1.0, #if args.classify_demo else args.class_sl_loss_weight,
                    reward_constant=args.reward_constant,
                    classify_thread=classify_thread,
                    sil_thread=sil_thread,
                    rollout_thread=rollout_thread,
                    exp_buffer=exp_buffer)

            if rollout_thread:
                local_network = GameACFFNetwork(
                    action_size, i, device=_device,
                    padding=args.padding, in_shape=input_shape)

                if args.sae_classify_demo:
                    # from class_network import AutoEncoderNetwork \
                    #     as PretrainedModelNetwork
                    logger.error("Classification type Not supported yet!")
                    assert False
                elif args.classify_demo:
                    from class_network import MultiClassNetwork \
                        as PretrainedModelNetwork
                else:
                    logger.error("Classification type Not supported yet!")
                    assert False

                local_pretrained_model = PretrainedModelNetwork(
                        pretrain_graph, action_size, i,
                        device=_device, padding=args.padding,
                        in_shape=input_shape, sae=args.sae_classify_demo,
                        tied_weights=args.class_tied_weights,
                        use_denoising=args.class_use_denoising,
                        noise_factor=args.class_noise_factor,
                        loss_function=args.class_loss_function,
                        use_slv=args.class_use_slv)

                from rollout_thread import RolloutThread as RolloutThread

                if args.unclipped_reward:
                    RolloutThread.reward_type = "RAW"
                elif args.log_scale_reward:
                    RolloutThread.reward_type = "LOG"
                else:
                    RolloutThread.reward_type = "CLIP"

                a3c_worker = RolloutThread(
                    thread_index=i, action_size=action_size, env_id=args.gym_env,
                    global_a3c=global_network, local_a3c=local_network,
                    global_pretrained_model=global_pretrained_model,
                    local_pretrained_model=local_pretrained_model,
                    max_global_time_step=args.max_time_step,
                    device=_device,
                    transformed_bellman = args.transformed_bellman)

        all_workers.append(a3c_worker)

    # prepare sessions
    sess = tf.Session(config=config)
    pretrain_sess = None
    if global_pretrained_model:
        pretrain_sess = tf.Session(config=config, graph=pretrain_graph)

    # initial a3c weights from pre-trained model
    if args.use_transfer:
        if args.transfer_folder is not None:
            transfer_folder = args.transfer_folder
        else:
            transfer_folder = 'results/pretrain_models/{}'.format(GYM_ENV_NAME)
            end_str = ''
            if args.use_mnih_2015:
                end_str += '_mnih2015'
            end_str += '_l2beta1E-04_oversample'  # TODO(make this an argument)
            transfer_folder += end_str

        transfer_folder = pathlib.Path(transfer_folder)
        transfer_folder /= 'transfer_model'

        if args.not_transfer_conv2:
            transfer_var_list = [
                global_network.W_conv1,
                global_network.b_conv1,
                ]

        elif (args.not_transfer_conv3 and args.use_mnih_2015):
            transfer_var_list = [
                global_network.W_conv1,
                global_network.b_conv1,
                global_network.W_conv2,
                global_network.b_conv2,
                ]

        elif args.not_transfer_fc1:
            transfer_var_list = [
                global_network.W_conv1,
                global_network.b_conv1,
                global_network.W_conv2,
                global_network.b_conv2,
                ]

            if args.use_mnih_2015:
                transfer_var_list += [
                    global_network.W_conv3,
                    global_network.b_conv3,
                    ]

        elif args.not_transfer_fc2:
            transfer_var_list = [
                global_network.W_conv1,
                global_network.b_conv1,
                global_network.W_conv2,
                global_network.b_conv2,
                global_network.W_fc1,
                global_network.b_fc1,
                ]

            if args.use_mnih_2015:
                transfer_var_list += [
                    global_network.W_conv3,
                    global_network.b_conv3,
                    ]

        else:
            transfer_var_list = [
                global_network.W_conv1,
                global_network.b_conv1,
                global_network.W_conv2,
                global_network.b_conv2,
                global_network.W_fc1,
                global_network.b_fc1,
                global_network.W_fc2,
                global_network.b_fc2,
                ]

            if args.use_mnih_2015:
                transfer_var_list += [
                    global_network.W_conv3,
                    global_network.b_conv3,
                    ]

            if '_slv' in str(transfer_folder):
                transfer_var_list += [
                    global_network.W_fc3,
                    global_network.b_fc3,
                    ]

        global_network.load_transfer_model(
            sess, folder=transfer_folder,
            not_transfer_fc2=args.not_transfer_fc2,
            not_transfer_fc1=args.not_transfer_fc1,
            not_transfer_conv3=(args.not_transfer_conv3
                                and args.use_mnih_2015),
            not_transfer_conv2=args.not_transfer_conv2,
            var_list=transfer_var_list,
            )

    # initial pretrained model
    if pretrain_sess:
        assert pretrained_model_folder is not None
        global_pretrained_model.load(
            pretrain_sess,
            pretrained_model_folder)

    def initialize_uninitialized(sess, model=None):
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

    if args.use_transfer:
        initialize_uninitialized(sess)
        if global_pretrained_model:
            initialize_uninitialized(pretrain_sess,
                                     global_pretrained_model)
        if local_pretrained_model:
            initialize_uninitialized(pretrain_sess,
                                     local_pretrained_model)
    else:
        sess.run(tf.global_variables_initializer())
        if global_pretrained_model:
            initialize_uninitialized(pretrain_sess,
                                     global_pretrained_model)
        if local_pretrained_model:
            initialize_uninitialized(pretrain_sess,
                                     local_pretrained_model)

    # summary writer for tensorboard
    # summary_op = tf.summary.merge_all()
    summ_file = 'scratch/cluster/yunshu/results/log/a3c/{}/'.format(GYM_ENV_NAME) + str(folder)[58:] # str(folder)[12:]
    summary_writer = tf.summary.FileWriter(summ_file, sess.graph)
    # summary_writer_class = None
    # if args.use_correction:
    #     summary_writer_class = tf.summary.FileWriter(summ_file,
    #                                                  pretrained_model_sess.graph)
    # init or load checkpoint with saver
    root_saver = tf.train.Saver(max_to_keep=1)
    # a3c
    saver = tf.train.Saver(max_to_keep=6)
    best_saver = tf.train.Saver(max_to_keep=1)
    # # classifier
    # class_root_saver = None
    # class_saver = None
    # best_class_saver = None
    # class_folder = str(folder) + '/classifier'
    # class_folder = pathlib.Path(class_folder)
    # if args.use_correction:
    #     class_root_saver = tf.train.Saver(max_to_keep=1)
    #     class_saver = tf.train.Saver(max_to_keep=6)
    #     best_class_saver = tf.train.Saver(max_to_keep=1)

    checkpoint = tf.train.get_checkpoint_state(str(folder))
    if checkpoint and checkpoint.model_checkpoint_path:
        root_saver.restore(sess, checkpoint.model_checkpoint_path)
        logger.info("checkpoint loaded:{}".format(
            checkpoint.model_checkpoint_path))
        tokens = checkpoint.model_checkpoint_path.split("-")
        # set global step
        global_t = int(tokens[-1])
        logger.info(">>> global step set: {}".format(global_t))

        # set wall time
        wall_t_fname = folder / 'wall_t.{}'.format(global_t)
        with wall_t_fname.open('r') as f:
            wall_t = float(f.read())

        # pretrain_t_file = folder / 'pretrain_global_t'
        # with pretrain_t_file.open('r') as f:
        #     pretrain_global_t = int(f.read())

        best_reward_file = folder / 'model_best/best_model_reward'
        with best_reward_file.open('r') as f:
            best_model_reward = float(f.read())
        reward_file = folder / '{}-a3c-rewards.pkl'.format(GYM_ENV_NAME)
        rewards = pickle.load(reward_file.open('rb'))

        # if args.use_correction:
        #     checkpoint_class = tf.train.get_checkpoint_state(str(class_folder))
        #     ckpt_path = checkpoint_class.model_checkpoint_path
        #     if checkpoint_class and ckpt_path:
        #         class_root_saver.restore(pretrained_model_folder, ckpt_path)
        #         logger.info("classifier checkpoint loaded:{}".format(ckpt_path))
        #         pretrain_global_t = int(ckpt_path.split("-")[-1])
        #         logger.info(">>> classifier global step: {}".format(pretrain_global_t))
        #         pretrain_t_file = folder / 'pretrain_global_t'
        #         with pretrain_t_file.open('r') as f:
        #             pretrain_global_t = int(f.read())
    else:
        logger.warning("Could not find old checkpoint")
        # set wall time
        wall_t = 0.0
        prepare_dir(folder, empty=True)
        prepare_dir(folder / 'model_checkpoints', empty=True)
        prepare_dir(folder / 'model_best', empty=True)
        prepare_dir(folder / 'frames', empty=True)
        # if args.use_correction:
        #     prepare_dir(class_folder / 'class_model_checkpoints', empty=True)
        #     prepare_dir(class_folder / 'class_model_best', empty=True)

    lock = threading.Lock()

    def next_t(current_t, freq):
        return np.ceil((current_t + 0.00001) / freq) * freq

    next_global_t = next_t(global_t, args.eval_freq)
    next_save_t = next_t(
        global_t, (args.max_time_step * args.max_time_step_fraction) // 5)
    step_t = 0

    last_temp_global_t = global_t
    ispretrain_markers = [False] * args.parallel_size

    def train_function(parallel_idx, th_ctr, ep_queue, net_updates,
                       class_updates, goodstate_queue, badstate_queue):
        nonlocal global_t, step_t, pretrain_global_t, pretrain_epoch, \
            rewards, class_rewards, lock, next_global_t, next_save_t, \
            last_temp_global_t, ispretrain_markers, shared_memory

        a3c_worker = all_workers[parallel_idx]
        a3c_worker.set_summary_writer(summary_writer)

        with lock:
            # Evaluate model before training
            if not stop_req and global_t == 0 and step_t == 0:
                rewards['eval'][step_t] = a3c_worker.testing(
                    sess, args.eval_max_steps, global_t, folder,
                    demo_memory_cam=demo_memory_cam, worker=all_workers[-1])
                # add testing for classifier in game
                if args.use_correction:
                    a3c_worker.test_game_classifier(global_t=global_t,
                                                    max_steps=args.eval_max_steps,
                                                    sess=pretrain_sess,
                                                    worker=all_workers[1])

                checkpt_file = folder / 'model_checkpoints'
                checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
                saver.save(sess, str(checkpt_file), global_step=global_t)
                save_best_model(rewards['eval'][global_t][0])

                step_t = 1

        # set start_time
        start_time = time.time() - wall_t
        a3c_worker.set_start_time(start_time)

        episode_end = True

        if a3c_worker.sil_thread:
            # TODO(add as command-line parameters later)
            sil_ctr = 0
            sil_interval = 0  # bigger number => slower SIL updates
            m_repeat = 4
            min_mem = args.batch_size * m_repeat
            sil_train_flag = args.load_memory and len(shared_memory) >= min_mem

        if a3c_worker.classify_thread:
            # TODO(add as command-line parameters later)
            classify_ctr = 0
            classify_interval = 10  # bigger number => slower classification updates
            classify_train_steps = args.class_train_steps
            # re-train if we have accumulated enough new samples
            # also consider the training time (about 6 min for 20k steps on GTX1060)
            classify_min_mem = args.batch_size * classify_train_steps * 0.1
            classify_train_flag = args.load_memory and \
                len(exp_buffer) >= classify_min_mem

        if a3c_worker.rollout_thread:
            rollout_ctr = 0
            # rollout_flag = False

        while True:
            if stop_req:
                return

            if global_t >= (args.max_time_step * args.max_time_step_fraction):
                if a3c_worker.sil_thread:
                    logger.info("SIL: # of updates: {}".format(sil_ctr))
                if a3c_worker.classify_thread:
                    logger.info("Classification: # of updates: {}".format(classify_ctr))
                if a3c_worker.rollout_thread:
                    logger.info("Rollout: # of corrections: {}".format(rollout_ctr))
                return

            if a3c_worker.sil_thread:
                if net_updates.qsize() >= sil_interval \
                   and len(shared_memory) >= min_mem:
                    sil_train_flag = True

                if sil_train_flag:
                    sil_train_flag = False
                    th_ctr.get()

                    train_out = a3c_worker.sil_train(
                        sess, global_t, shared_memory, sil_ctr, m=m_repeat)
                    sil_ctr, goodstate, badstate = train_out

                    th_ctr.put(1)

                    with net_updates.mutex:
                        net_updates.queue.clear()

                    if args.use_correction:
                        class_updates.put(1)
                        # add rollout new_ep to ep_queue
                        while not goodstate.empty():
                            goodstate_queue.put(goodstate.get())
                        while not badstate.empty():
                            if badstate_queue.full():
                                badstate_queue.get()
                            badstate_queue.put(badstate.get())

                    with goodstate.mutex:
                        goodstate.queue.clear()
                    with badstate.mutex:
                        badstate.queue.clear()

                    assert goodstate.empty()
                    assert badstate.empty()

                    if sil_ctr % 100 == 0:
                        if args.use_correction:
                            log_data = (sil_ctr, len(shared_memory),
                                        goodstate_queue.qsize(),
                                        badstate_queue.qsize())
                            logger.info("SIL: sil_ctr={}"
                                        " sil_memory_size={}"
                                        " #good_states={}"
                                        " #bad_states={}".format(*log_data))
                        else:
                            log_data = (sil_ctr, len(shared_memory))
                            logger.info("SIL: sil_ctr={}"
                                        " sil_memory_size={}".format(*log_data))

                # Adding episodes to SIL memory is centralize to ensure
                # sampling and updating of priorities does not become a problem
                # since we add new episodes to SIL at once and during
                # SIL training it is guaranteed that SIL memory is untouched.
                max = args.parallel_size
                # if args.use_sil:
                #     max = max - 1
                # if args.use_correction:
                #     max = max - 1

                while not ep_queue.empty():
                    data = ep_queue.get()
                    a3c_worker.episode.set_data(*data)
                    shared_memory.extend(a3c_worker.episode)
                    a3c_worker.episode.reset()
                    max -= 1
                    # This ensures that SIL has a chance to train
                    if max <= 0:
                        break

                # # at sil_interval=0, this will never be executed,
                # # this is considered fast SIL (no intervals between updates)
                # if not sil_train_flag and len(shared_memory) >= min_mem:
                #     # No training, just updating priorities
                #     a3c_worker.sil_update_priorities(
                #         sess, shared_memory, m=m_repeat)

                diff_global_t = 0


            elif a3c_worker.classify_thread:
                # print(class_updates.qsize())
                if class_updates.qsize() >= classify_interval: #and len(exp_buffer) >= classify_min_mem:
                    classify_train_flag = True
                    count = goodstate_queue.qsize()
                    logger.info("Filling exp_buffer with {} new samples".format(count))
                    # Adding episodes to exp_buffer is centralize in classify_thread
                    # we add new episodes to exp_buffer at once when classification flag is true
                    # but also note that SIL thread will be adding new ep simoutanously
                    # so we should control the max # ep to add
                    # otherwise might be adding forever.
                    while not goodstate_queue.empty():
                        data = goodstate_queue.get()
                        s, fs, a, _return = data
                        exp_buffer.extend_one(s, fs, a, _return)
                        count -= 1
                        if count <= 0:
                            break

                if classify_train_flag:
                    classify_train_flag = False
                    logger.info("Start training classifier...")
                    th_ctr.get()

                    classify_ctr = a3c_worker.train(
                        pretrain_sess, exp_buffer, classify_ctr)

                    th_ctr.put(1)
                    logger.info("Finish training classifier")

                    with class_updates.mutex:
                        class_updates.queue.clear()

                diff_global_t = 0


            elif a3c_worker.rollout_thread:
                th_ctr.get()
                diff_global_t = 0

                # badstate_size = badstate_queue.qsize()
                tmp_queue = Queue()
                # for _ in range(badstate_size):
                tmp_queue.put(badstate_queue.get())


                # while badstate_size > 0:
                if not tmp_queue.empty():
                    train_out = a3c_worker.rollout(sess, pretrain_sess,
                                                   global_t, tmp_queue,
                                                   rollout_ctr)
                    tmp_queue, diff_global_t, \
                        episode_end, part_end, rollout_ctr = train_out
                    # diff_global_t += diff_t

                    # should always part_end
                    # put new trajectory into ep_queue, which goes to shared_memory later
                    assert part_end
                    # badstate_size -= 1

                    if part_end:
                        ep_queue.put(a3c_worker.episode.get_data())
                        a3c_worker.episode.reset()

                th_ctr.put(1)

                assert tmp_queue.empty()
                del tmp_queue


            else:
                th_ctr.get()

                train_out = a3c_worker.train(sess, global_t, rewards)
                diff_global_t, episode_end, part_end = train_out

                th_ctr.put(1)

                if args.use_sil:
                    net_updates.put(1)
                    if part_end:
                        ep_queue.put(a3c_worker.episode.get_data())
                        a3c_worker.episode.reset()

            with lock:
                global_t += diff_global_t

                if global_t > next_global_t:
                    next_global_t = next_t(global_t, args.eval_freq)
                    step_t = int(next_global_t - args.eval_freq)

                    # wait for all threads to finish before testing
                    while not stop_req and th_ctr.qsize() < len(all_workers):
                        time.sleep(0.01)

                    step_t = int(next_global_t - args.eval_freq)

                    rewards['eval'][step_t] = a3c_worker.testing(
                            sess, args.eval_max_steps, step_t, folder,
                            demo_memory_cam=demo_memory_cam, worker=all_workers[-1])
                    save_best_model(rewards['eval'][step_t][0])

                    # add testing for classifier in game
                    if args.use_correction:
                        a3c_worker.test_game_classifier(global_t=step_t,
                                                        max_steps=args.eval_max_steps,
                                                        sess=pretrain_sess,
                                                        worker=all_workers[1])
                                                        #model=pretrained_model)


            if global_t > next_save_t:
                freq = (args.max_time_step * args.max_time_step_fraction)
                freq = freq // 5
                next_save_t = next_t(global_t, freq)
                # save a3c
                checkpt_file = folder / 'model_checkpoints'
                checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
                saver.save(sess, str(checkpt_file), global_step=global_t,
                           write_meta_graph=False)
                # # save classifier
                # if args.use_correction and class_saver is not None:
                #     checkpt_file = folder / 'class_model_checkpoints'
                #     checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
                #     class_saver.save(pretrained_model_sess, str(checkpt_file),
                #         global_step=global_t,
                #         write_meta_graph=False)

    def signal_handler(signal, frame):
        nonlocal stop_req
        logger.info('You pressed Ctrl+C!')
        stop_req = True

        if stop_req and global_t == 0:
            sys.exit(1)

    def save_best_model(test_reward):
        nonlocal best_model_reward
        if test_reward > best_model_reward:
            best_model_reward = test_reward
            best_reward_file = folder / 'model_best/best_model_reward'

            with best_reward_file.open('w') as f:
                f.write(str(best_model_reward))

            best_checkpt_file = folder / 'model_best'
            best_checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
            best_saver.save(sess, str(best_checkpt_file))

    # def class_save_best_model(test_reward):
    #     """Save best classifier model's parameters and reward to file.
    #
    #     Keyword arguments:
    #     test_reward -- testing total average reward
    #     best_saver -- tf saver object
    #     sess -- tf session
    #     """
    #     nonlocal class_best_model_reward
    #     if test_reward > class_best_model_reward:
    #         class_best_model_reward = test_reward
    #         best_class_reward_file = class_folder / 'class_model_best/best_model_reward'
    #         with best_class_reward_file.open('w') as f:
    #             f.write(str(class_best_model_reward))
    #
    #         best_checkpt_file = class_folder / 'class_model_best'
    #         best_checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
    #         best_
    #
    #         best_class_saver.save(pretrained_model_sess, str(best_checkpt_file))

    train_threads = []
    th_ctr = Queue()
    for i in range(args.parallel_size):
        th_ctr.put(1)
    episodes_queue = None
    net_updates = None
    class_updates = None
    goodstate_queue = None
    badstate_queue = None
    if args.use_sil:
        episodes_queue = Queue()
        net_updates = Queue()
    if args.use_correction:
        class_updates = Queue()
        goodstate_queue = Queue()
        badstate_queue = Queue(maxsize=args.memory_length//2)
    for i in range(args.parallel_size):
        worker_thread = threading.Thread(
            target=train_function,
            args=(i, th_ctr, episodes_queue, net_updates, class_updates,
                goodstate_queue, badstate_queue,))
        train_threads.append(worker_thread)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # set start time
    start_time = time.time() - wall_t

    for t in train_threads:
        t.start()

    print('Press Ctrl+C to stop')

    for t in train_threads:
        t.join()

    logger.info('Now saving data. Please wait')

    # write wall time
    wall_t = time.time() - start_time
    wall_t_fname = folder / 'wall_t.{}'.format(global_t)
    with wall_t_fname.open('w') as f:
        f.write(str(wall_t))


    checkpoint_file = str(folder / '{}_checkpoint_a3c'.format(GYM_ENV_NAME))
    root_saver.save(sess, checkpoint_file, global_step=global_t)

    # if args.use_correction:
    #     pretrain_gt_fname = class_folder / 'pretrain_global_t'
    #     with pretrain_gt_fname.open('w') as f:
    #         f.write(str(pretrain_global_t))
    #     checkpoint_file_class = str(class_folder / '{}_checkpoint_class'.format(GYM_ENV_NAME))
    #     class_root_saver.save(pretrained_model_sess, checkpoint_file,
    #         global_step=pretrain_global_t)


    reward_fname = folder / '{}-a3c-rewards.pkl'.format(GYM_ENV_NAME)
    pickle.dump(rewards, reward_fname.open('wb'), pickle.HIGHEST_PROTOCOL)
    logger.info('Data saved!')

    sess.close()
    if pretrain_sess:
        pretrain_sess.close()
