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
    --pretrained-model-folder=<>
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
import pandas as pd
import math

from threading import Event, Thread
from common_worker import CommonWorker
from a3c_training_thread import A3CTrainingThread
from sil_training_thread import SILTrainingThread
from rollout_thread import RolloutThread
from classifier_thread import ClassifyDemo as ClassifierThread
from class_network import MultiClassNetwork as PretrainedModelNetwork
from common.game_state import GameState
from common.util import prepare_dir
from game_ac_network import GameACFFNetwork
from queue import Queue, PriorityQueue
from sil_memory import SILReplayMemory
from copy import deepcopy
from setup_functions import *


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
    GAME_NAME = args.gym_env.replace('NoFrameskip-v4','')

    # setup folder name and path to folder
    folder = pathlib.Path(setup_folder(args, GYM_ENV_NAME))

    # setup demo folder path
    demo_memory_folder = None
    if args.load_memory or args.load_demo_cam:
        path = setup_demofolder(args.demo_memory_folder, GYM_ENV_NAME)
        demo_memory_folder = pathlib.Path(path)

    # setup by loading demo memory
    demo_memory = setup_demo_memory(args.load_memory, args.use_sil,
                                    args.demo_ids, demo_memory_folder)
    demo_memory_cam = setup_demo_memory_cam(args.load_demo_cam,
        args.demo_cam_id, demo_memory_folder)
    if demo_memory_cam is not None:
        logger.info("loaded demo {} for testing CAM".format(args.demo_cam_id))

    # setup GPU options
    import tensorflow as tf
    gpu_options = setup_gpu(tf, args.use_gpu, args.gpu_fraction)

    ######################################################
    # setup default device
    device = "/cpu:0"

    global_t = 0
    rewards = {'train': {}, 'eval': {}}
    best_model_reward = -(sys.maxsize)
    if args.load_pretrained_model:
        class_best_model_reward = -(sys.maxsize)
        class_rewards = {'class_eval': {}}

    sil_dict = {"sil_ctr":{}, # total number of SIL updates
                "sil_a3c_sampled":{}, # total number of a3c experience sampled during SIL
                "sil_a3c_used":{}, # total number of a3c experience used during SIL (i.e., pass max op)
                "sil_a3c_sampled_return":{},# the average return of sampled experiences from buffer D
                "sil_a3c_used_return":{}, # the average return of used experiences for butter D
                "sil_a3c_used_adv":{}, # the average advantage of used experiences for butter D (should always be pos)
                "sil_rollout_sampled":{}, # total number of rollout experience sampled during SIL
                "sil_rollout_used":{}, # total number of rollout experience used during SIL (i.e., pass max op)
                "sil_rollout_sampled_return":{}, # the average return of sampled experiences from buffer D
                "sil_rollout_used_return":{}, # the average return of sampled experiences from buffer R
                "sil_rollout_used_adv":{}, # the average advantage of used experiences for butter R (should always be pos)
                "sil_old_sampled":{}, # total number of older experience sampled; once an experience is refreshed, its older ver should never be used again)
                "sil_old_used":{}} # total number of older exprience used (i.e., pass max op)
    sil_ctr, sil_a3c_sampled, sil_a3c_used, sil_a3c_sampled_return, sil_a3c_used_return = 0, 0, 0, 0, 0
    sil_a3c_used_adv = 0
    sil_rollout_sampled, sil_rollout_used, sil_rollout_sampled_return, sil_rollout_used_return = 0, 0, 0, 0
    sil_rollout_used_adv = 0
    sil_old_sampled, sil_old_used = 0, 0

    rollout_dict = {"rollout_ctr": {}, # total number of rollout performed (i.e., refresh freq)
                    "rollout_added_ctr":{}, # total number of successful rollout (i.e., freq of Gnew > G, also is number of rollout updates)
                    "rollout_steps": {}, # total number of rollout steps (50M - rollout_steps = a3c_steps)
                    "rollout_sample_used":{}, # total number of sample used for rollout update (each traj is of diff length, what's the avg)
                    "rollout_sample_used_adv":{}, # the advantage of samples used for rollout update (each traj is of diff length, what's the avg)
                    "rollout_new_return":{}, # avg new return value for rollout (show if addall, this will be lower)
                    "rollout_old_return":{}} # avg old return value (show old_return is lower than new_return)
    rollout_ctr, rollout_added_ctr, rollout_sample_used, rollout_steps = 0, 0, 0, 0
    rollout_sample_used_adv = 0
    rollout_new_return = 0 # this records the total, avg = this / rollout_added_ctr
    rollout_old_return = 0 # this records the total, avg = this / rollout_added_ctr

    a3c_dict = {"a3c_ctr":{}, # total number of A3C updates
                "a3c_sample_used":{},# total number of samples used for A3C updates (each traj is diff length)
                "a3c_sample_used_return":{},# the return of samples used for A3C updates, compare with the returns in SIL
                "a3c_sample_used_adv":{}} # the advantage of samples used for A3C updates, compare with the returns in SIL
    a3c_ctr, a3c_sample_used, a3c_sample_used_return, a3c_sample_used_adv = 0, 0, 0, 0

    # set up file names
    reward_fname = folder / '{}-a3c-rewards.pkl'.format(GYM_ENV_NAME)
    sil_fname = folder / '{}-a3c-dict-sil.pkl'.format(GYM_ENV_NAME)
    rollout_fname = folder / '{}-a3c-dict-rollout.pkl'.format(GYM_ENV_NAME)
    a3c_fname = folder / '{}-a3c-dict-a3c.pkl'.format(GYM_ENV_NAME)
    if args.load_pretrained_model:
        class_reward_fname = folder / '{}-class-rewards.pkl'.format(GYM_ENV_NAME)

    sharedmem_fname = folder / '{}-sharedmem.pkl'.format(GYM_ENV_NAME)
    sharedmem_params_fname = folder / '{}-sharedmem-params.pkl'.format(GYM_ENV_NAME)
    sharedmem_trees_fname = folder / '{}-sharedmem-trees.pkl'.format(GYM_ENV_NAME)

    rolloutmem_fname = folder / '{}-rolloutmem.pkl'.format(GYM_ENV_NAME)
    rolloutmem_params_fname = folder / '{}-rolloutmem-params.pkl'.format(GYM_ENV_NAME)
    rolloutmem_trees_fname = folder / '{}-rolloutmem-trees.pkl'.format(GYM_ENV_NAME)

    stop_req = False
    a3clog = False

    game_state = GameState(env_id=args.gym_env)
    action_size = game_state.env.action_space.n
    game_state.close()
    del game_state.env
    del game_state

    input_shape = (args.input_shape, args.input_shape, 4)
    #######################################################
    # setup global A3C
    GameACFFNetwork.use_mnih_2015 = args.use_mnih_2015
    global_network = GameACFFNetwork(
        action_size, -1, device, padding=args.padding,
        in_shape=input_shape)
    logger.info('A3C Initial Learning Rate={}'.format(args.initial_learn_rate))

    # setup pretrained model
    global_pretrained_model = None
    local_pretrained_model = None
    pretrain_graph = None

    # if use pretrained model as (a part of) rollout
    # then must load pretrained model
    # otherwise, don't load model (for saving mem)
    if args.use_rollout and args.nstep_bc > 0:
        assert args.load_pretrained_model
    else:
        assert not args.load_pretrained_model

    if args.load_pretrained_model:
        pretrain_graph, global_pretrained_model = setup_pretrained_model(tf,
            args, action_size, input_shape,
            device="/gpu:0" if args.use_gpu else device)
        assert global_pretrained_model is not None
        assert pretrain_graph is not None
        if args.train_classifier:
            logger.info("Classifier optimizer: {}".format(
                'RMSPropOptimizer' if args.class_optimizer == 'rms' else 'AdamOptimizer'))
            logger.info("Classifier learning_rate: {}".format(args.class_learn_rate))
            logger.info("Classifier epsilon: {}".format(args.class_opt_epsilon))
            if args.class_optimizer == 'rms':
                logger.info("\trms decay: {}".format(args.rmsp_alpha))
            class_ae_opt, class_opt = setup_class_optimizer(tf, args)

    time.sleep(2.0)

    # setup experience memory
    shared_memory = None
    exp_buffer = None
    rollout_buffer = None
    if args.use_sil:
        shared_memory = SILReplayMemory(
            action_size, max_len=args.memory_length, gamma=args.gamma,
            clip=False if args.unclipped_reward else True,
            height=input_shape[0], width=input_shape[1],
            phi_length=input_shape[2], priority=args.priority_memory,
            reward_constant=args.reward_constant)

        if args.use_rollout and not args.one_buffer:
            rollout_buffer = SILReplayMemory(
                action_size, max_len=args.memory_length, gamma=args.gamma,
                clip=False if args.unclipped_reward else True,
                height=input_shape[0], width=input_shape[1],
                phi_length=input_shape[2], priority=args.priority_memory,
                reward_constant=args.reward_constant)

        if args.train_classifier:
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
            # if args.use_rollout:
            #     temp_memory2 = SILReplayMemory(
            #         action_size, max_len=args.memory_length, gamma=args.gamma,
            #         clip=False if args.unclipped_reward else True,
            #         height=input_shape[0], width=input_shape[1],
            #         phi_length=input_shape[2], priority=False,
            #         reward_constant=args.reward_constant)
            if args.train_classifier:
                temp_memory3 = SILReplayMemory(
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
                    # if args.use_rollout:
                    #     temp_memory2.add_item(s0, fs0, a0, r1, t1)
                    if args.train_classifier:
                        temp_memory3.add_item(s0, fs0, a0, r1, t1)

                    if t1:  # terminal
                        shared_memory.fs_size = len(fs0)
                        shared_memory.extend(temp_memory)
                        if args.use_rollout and not args.one_buffer:
                            rollout_buffer.fs_size = len(fs0)
                        #     rollout_buffer.extend(temp_memory2)
                        if args.train_classifier:
                            exp_buffer.fs_size = len(fs0)
                            exp_buffer.extend(temp_memory3)

                if len(temp_memory) > 0:
                    logger.warning("Disregard {} states in"
                                   " demo_memory {}".format(
                                    len(temp_memory), idx))
                    temp_memory.reset()
                # if args.use_rollout:
                #     if len(temp_memory2) > 0:
                #         logger.warning("Disregard {} states in"
                #                        " demo_memory {}".format(
                #                         len(temp_memory2), idx))
                #         temp_memory2.reset()
                if args.train_classifier:
                    if len(temp_memory3) > 0:
                        logger.warning("Disregard {} states in"
                                       " demo_memory {}".format(
                                        len(temp_memory3), idx))
                        temp_memory3.reset()

            del temp_memory
            if args.use_rollout and not args.one_buffer:
                pass
                # del temp_memory2
            if args.train_classifier:
                del temp_memory3

        # log memory information
        shared_memory.log()
        if args.use_rollout and not args.one_buffer:
            rollout_buffer.log()
        if args.train_classifier:
            exp_buffer.log()

    ############## Setup Thread Workers BEGIN ################
    # one sil thread, one classifier thread, one rollout thread, others are a3c
    if args.use_sil and args.use_rollout and args.train_classifier:
        assert args.parallel_size >= 4

    startIndex = 0
    all_workers = []

    # a3c and sil learning rate and optimizer
    learning_rate_input = tf.placeholder(tf.float32, shape=(), name="opt_lr")
    grad_applier = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate_input,
        decay=args.rmsp_alpha,
        epsilon=args.rmsp_epsilon)

    setup_common_worker(CommonWorker, args, action_size)

    # setup SIL worker
    sil_worker = None
    sil_addmem_thread = -1
    if args.use_sil:
        _device = "/gpu:0" if (args.use_gpu and args.num_sil_worker==1) else device
        sil_addmem_thread = startIndex
        for i in range(args.num_sil_worker):
            sil_network = GameACFFNetwork(
                action_size, startIndex, device=_device, #"/gpu:0" if args.use_gpu else device,
                padding=args.padding, in_shape=input_shape)

            sil_worker = SILTrainingThread(startIndex, global_network, sil_network,
                args.initial_learn_rate,
                learning_rate_input,
                grad_applier, device=_device, #"/gpu:0" if args.use_gpu else device,
                batch_size=args.batch_size,
                use_rollout=args.use_rollout,
                rollout_sample_proportion=args.rollout_sample_proportion,
                train_classifier=args.train_classifier,
                use_sil_neg=args.use_sil_neg,
                one_buffer=args.one_buffer)

            all_workers.append(sil_worker)
            startIndex += 1

    # setup rollout worker
    rollout_worker = None
    if args.use_rollout:
        _device = "/gpu:0" if (args.use_gpu and args.num_rollout_worker==1) else device
        for i in range(args.num_rollout_worker):
            rollout_network = GameACFFNetwork(
                action_size, startIndex, device=_device, #"/gpu:0" if args.use_gpu else device,
                padding=args.padding, in_shape=input_shape)

            rollout_local_pretrained_model = None
            if args.nstep_bc >0:
                rollout_local_pretrained_model = PretrainedModelNetwork(
                    pretrain_graph, action_size, startIndex,
                    padding=args.padding,
                    in_shape=input_shape, sae=args.sae_classify_demo,
                    tied_weights=args.class_tied_weights,
                    use_denoising=args.class_use_denoising,
                    noise_factor=args.class_noise_factor,
                    loss_function=args.class_loss_function,
                    use_slv=args.class_use_slv,
                    device=_device) #"/gpu:0" if args.use_gpu else device)

            rollout_worker = RolloutThread(
                thread_index=startIndex, action_size=action_size, env_id=args.gym_env,
                global_a3c=global_network, local_a3c=rollout_network,
                update_in_rollout=args.update_in_rollout, nstep_bc=args.nstep_bc,
                global_pretrained_model=global_pretrained_model,
                local_pretrained_model=rollout_local_pretrained_model,
                transformed_bellman = args.transformed_bellman,
                device=_device, #"/gpu:0" if args.use_gpu else device,
                entropy_beta=args.entropy_beta, clip_norm=args.grad_norm_clip,
                grad_applier=grad_applier,
                initial_learn_rate=args.initial_learn_rate,
                learning_rate_input=learning_rate_input)

            all_workers.append(rollout_worker)
            startIndex += 1

    # setup classifier training worker
    classifier_worker = None
    classifier_index = None
    if args.load_pretrained_model and args.train_classifier:
        classifier_worker = ClassifierThread(
            tf=tf, net=global_pretrained_model,
            thread_index=startIndex,
            game_name=args.gym_env,
            train_max_steps=int(args.class_train_steps),
            batch_size=args.batch_size,
            ae_grad_applier=class_ae_opt,
            grad_applier=class_opt,
            eval_freq=args.eval_freq,
            clip_norm=args.grad_norm_clip,
            game_state=GameState(env_id=args.gym_env, no_op_max=30),
            sampling_type=args.class_sample_type,
            sl_loss_weight=1.0, #if args.classify_demo else args.class_sl_loss_weight,
            # reward_constant=args.reward_constant,
            exp_buffer=exp_buffer)
        # exclude testing samples from classifier
        exp_buffer.states = [element for i, element in enumerate(exp_buffer.states) \
            if i not in classifier_worker.test_index_list]
        exp_buffer.actions = [element for i, element in enumerate(exp_buffer.actions) \
            if i not in classifier_worker.test_index_list]
        exp_buffer.rewards = [element for i, element in enumerate(exp_buffer.rewards) \
            if i not in classifier_worker.test_index_list]
        exp_buffer.terminal = [element for i, element in enumerate(exp_buffer.terminal) \
            if i not in classifier_worker.test_index_list]
        exp_buffer.returns = [element for i, element in enumerate(exp_buffer.returns) \
            if i not in classifier_worker.test_index_list]
        exp_buffer.fullstates = [element for i, element in enumerate(exp_buffer.fullstates) \
            if i not in classifier_worker.test_index_list]

        all_workers.append(classifier_worker)
        classifier_index = startIndex

        startIndex += 1

    # setup a3c workers
    setup_a3c_worker(A3CTrainingThread, args, startIndex)

    for i in range(startIndex, args.parallel_size):
        local_network = GameACFFNetwork(
            action_size, i, device="/cpu:0",
            padding=args.padding,
            in_shape=input_shape)

        a3c_worker = A3CTrainingThread(
            i, global_network, local_network,
            args.initial_learn_rate, learning_rate_input, grad_applier,
            device="/cpu:0", no_op_max=30)

        all_workers.append(a3c_worker)
    ############## Setup Thread Workers END ################

    # setup config for tensorflow
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
        allow_soft_placement=True)

    # prepare sessions
    sess = tf.Session(config=config)
    pretrain_sess = None
    if global_pretrained_model:
        pretrain_sess = tf.Session(config=config, graph=pretrain_graph)

    # initial a3c weights from pre-trained model
    if args.use_transfer:
        transfer_folder = pathlib.Path(setup_transfer_folder(args, GYM_ENV_NAME))
        transfer_folder /= 'transfer_model'

        transfer_vars = setup_transfer_vars(args, global_network, transfer_folder)

        global_network.load_transfer_model(
            sess, folder=transfer_folder,
            not_transfer_fc2=args.not_transfer_fc2,
            not_transfer_fc1=args.not_transfer_fc1,
            not_transfer_conv3=(args.not_transfer_conv3
                                and args.use_mnih_2015),
            not_transfer_conv2=args.not_transfer_conv2,
            var_list=transfer_vars,
            )

    # initial pretrained model
    if pretrain_sess:
        assert args.pretrained_model_folder is not None
        global_pretrained_model.load(
            pretrain_sess,
            args.pretrained_model_folder)

    if args.use_transfer:
        initialize_uninitialized(tf, sess)
        if global_pretrained_model:
            initialize_uninitialized(tf, pretrain_sess,
                                     global_pretrained_model)
        if local_pretrained_model:
            initialize_uninitialized(tf, pretrain_sess,
                                     local_pretrained_model)
    else:
        sess.run(tf.global_variables_initializer())
        if global_pretrained_model:
            initialize_uninitialized(tf, pretrain_sess,
                                     global_pretrained_model)
        if local_pretrained_model:
            initialize_uninitialized(tf, pretrain_sess,
                                     local_pretrained_model)

    # summary writer for tensorboard
    # summary_op = tf.summary.merge_all()
    summ_file = args.save_to+'log/a3c/{}/'.format(GYM_ENV_NAME) + str(folder)[58:] # str(folder)[12:]
    summary_writer = tf.summary.FileWriter(summ_file, sess.graph)

    # init or load checkpoint with saver
    root_saver = tf.train.Saver(max_to_keep=1)
    saver = tf.train.Saver(max_to_keep=3)
    best_saver = tf.train.Saver(max_to_keep=1)

    checkpoint = tf.train.get_checkpoint_state(str(folder)+'/model_checkpoints')
    if checkpoint and checkpoint.model_checkpoint_path:
        root_saver.restore(sess, checkpoint.model_checkpoint_path)
        logger.info("checkpoint loaded:{}".format(
            checkpoint.model_checkpoint_path))
        tokens = checkpoint.model_checkpoint_path.split("-")
        # set global step
        global_t = int(tokens[-1])
        logger.info(">>> global step set: {}".format(global_t))

        tmp_t = (global_t // args.eval_freq) * args.eval_freq
        logger.info(">>> tmp_t: {}".format(tmp_t))

        # set wall time
        wall_t = 0.
        # wall_t_fname = folder / 'model_checkpoints/wall_t.{}'.format(global_t)
        # with wall_t_fname.open('r') as f:
        #     wall_t = float(f.read())

        # pretrain_t_file = folder / 'pretrain_global_t'
        # with pretrain_t_file.open('r') as f:
        #     pretrain_global_t = int(f.read())

        # set up reward files
        best_reward_file = folder / 'model_best/best_model_reward'
        with best_reward_file.open('r') as f:
            best_model_reward = float(f.read())

        rewards = pickle.load(reward_fname.open('rb'))
        logger.info(">>> restored: rewards")

        sil_dict = pickle.load(sil_fname.open('rb'))
        sil_ctr = sil_dict['sil_ctr'][tmp_t]
        sil_a3c_sampled = sil_dict['sil_a3c_sampled'][tmp_t]
        sil_a3c_used = sil_dict['sil_a3c_used'][tmp_t]
        sil_a3c_sampled_return = sil_dict['sil_a3c_sampled_return'][tmp_t]
        sil_a3c_used_return = sil_dict['sil_a3c_used_return'][tmp_t]
        sil_a3c_used_adv = sil_dict['sil_a3c_used_adv'][tmp_t]
        sil_rollout_sampled = sil_dict['sil_rollout_sampled'][tmp_t]
        sil_rollout_used = sil_dict['sil_rollout_used'][tmp_t]
        sil_rollout_sampled_return = sil_dict['sil_rollout_sampled_return'][tmp_t]
        sil_rollout_used_return = sil_dict['sil_rollout_used_return'][tmp_t]
        sil_rollout_used_adv = sil_dict['sil_rollout_used_adv'][tmp_t]
        sil_old_sampled = sil_dict['sil_old_sampled'][tmp_t]
        sil_old_used = sil_dict['sil_old_used'][tmp_t]
        logger.info(">>> restored: sil_dict")

        rollout_dict = pickle.load(rollout_fname.open('rb'))
        rollout_ctr = rollout_dict['rollout_ctr'][tmp_t]
        rollout_added_ctr = rollout_dict['rollout_added_ctr'][tmp_t]
        rollout_steps = rollout_dict['rollout_steps'][tmp_t]
        rollout_sample_used = rollout_dict['rollout_sample_used'][tmp_t]
        rollout_sample_used_adv = rollout_dict['rollout_sample_used_adv'][tmp_t]
        rollout_new_return = rollout_dict['rollout_new_return'][tmp_t]
        rollout_old_return = rollout_dict['rollout_old_return'][tmp_t]
        logger.info(">>> restored: rollout_dict")

        a3c_dict = pickle.load(a3c_fname.open('rb'))
        a3c_ctr = a3c_dict['a3c_ctr'][tmp_t]
        a3c_sample_used = a3c_dict['a3c_sample_used'][tmp_t]
        a3c_sample_used_return = a3c_dict['a3c_sample_used_return'][tmp_t]
        a3c_sample_used_adv = a3c_dict['a3c_sample_used_adv'][tmp_t]
        logger.info(">>> restored: a3c_dict")

        if args.load_pretrained_model:
            class_reward_file = folder / '{}-class-rewards.pkl'.format(GYM_ENV_NAME)
            class_rewards = pickle.load(class_reward_file.open('rb'))

        # set up replay buffers
        if args.checkpoint_buffer:
            if args.use_sil and args.priority_memory:
                shared_memory = restore_buffer(sharedmem_fname, shared_memory)
                shared_memory = restore_buffer_trees(sharedmem_trees_fname, shared_memory)
                shared_memory = restore_buffer_params(sharedmem_params_fname, shared_memory)
                shared_memory.log()

                if args.use_rollout and not args.one_buffer:
                    rollout_buffer = restore_buffer(rolloutmem_fname, rollout_buffer)
                    rollout_buffer = restore_buffer_trees(rolloutmem_trees_fname, rollout_buffer)
                    rollout_buffer = restore_buffer_params(rolloutmem_params_fname, rollout_buffer)
                    rollout_buffer.log()

    else:
        logger.warning("Could not find old checkpoint")
        # set wall time
        wall_t = 0.0
        prepare_dir(folder, empty=True)
        prepare_dir(folder / 'model_checkpoints', empty=True)
        prepare_dir(folder / 'model_best', empty=True)
        prepare_dir(folder / 'frames', empty=True)
        # prepare_dir(folder / 'rollout', empty=True)

    lock = threading.Lock()
    sil_lock = threading.Lock()
    rollout_lock = threading.Lock()

    def next_t(current_t, freq):
        return np.ceil((current_t + 0.00001) / freq) * freq

    next_global_t = next_t(global_t, args.eval_freq)
    next_save_t = next_t(
        global_t, args.eval_freq*args.checkpoint_freq) #(args.max_time_step * args.max_time_step_fraction) // 5)

    step_t = 0

    # last_temp_global_t = global_t
    # ispretrain_markers = [False] * args.parallel_size

    # rollout_sample_proportion = args.rollout_sample_proportion
    _stop_rollout = False
    last_reward = -(sys.maxsize)
    class_last_reward = -(sys.maxsize)

    def train_function(parallel_idx, th_ctr, ep_queue, net_updates, class_updates, \
                       goodstate_queue, badstate_queue, badstate_list, \
                       th_sil_ctr):
        nonlocal global_t, step_t, rewards, class_rewards, lock, sil_lock, rollout_lock, \
                 next_save_t, next_global_t, a3clog
        nonlocal shared_memory, exp_buffer, rollout_buffer
        nonlocal _stop_rollout, last_reward, class_last_reward #rollout_sample_proportion,
        nonlocal sil_dict, sil_ctr, sil_a3c_sampled, sil_a3c_used, \
                 sil_a3c_sampled_return, sil_a3c_used_return, sil_a3c_used_adv,\
                 sil_rollout_sampled, sil_rollout_used, \
                 sil_rollout_sampled_return, sil_rollout_used_return, sil_rollout_used_adv,\
                 sil_old_sampled, sil_old_used
        nonlocal rollout_dict, rollout_ctr, rollout_added_ctr, rollout_sample_used,\
                 rollout_sample_used_adv,\
                 rollout_new_return, rollout_old_return, rollout_steps
        nonlocal a3c_dict, a3c_ctr, a3c_sample_used, a3c_sample_used_return, a3c_sample_used_adv
        # ispretrain_markers, # last_temp_global_t,\

        parallel_worker = all_workers[parallel_idx]
        parallel_worker.set_summary_writer(summary_writer)

        with lock:
            # Evaluate model before training
            if not stop_req and global_t == 0 and step_t == 0:
                # # testing for 100 ep
                # TODO: move to testing script
                # _,_,_,reward_list = parallel_worker.testing_ep(sess, args.eval_max_steps,
                #     global_t, folder, demo_memory_cam=demo_memory_cam, worker=all_workers[-1])
                # reward_fname = folder / '{}-100ep-rewards.pkl'.format(GYM_ENV_NAME)
                # pickle.dump(reward_list, reward_fname.open('wb'), pickle.HIGHEST_PROTOCOL)
                # # read checkpoint directly from transfer_folder when testing episodically
                # checkpoint = tf.train.get_checkpoint_state(str(args.transfer_folder))
                # #
                rewards['eval'][step_t] = parallel_worker.testing(
                    sess, args.eval_max_steps, global_t, folder,
                    demo_memory_cam=demo_memory_cam, worker=all_workers[-1])
                last_reward = rewards['eval'][step_t][0]
                # testing classifier in game (retrain classifier)
                if args.train_classifier:
                    assert classifier_index is not None
                    class_rewards['class_eval'][step_t] = \
                        parallel_worker.test_retrain_classifier(global_t=global_t,
                                                    max_steps=args.eval_max_steps,
                                                    sess=pretrain_sess,
                                                    worker=all_workers[classifier_index])
                    class_last_reward = class_rewards['class_eval'][step_t][0]
                # testing classifier in game (fix classifier)
                elif args.load_pretrained_model:
                    assert pretrain_sess is not None
                    assert global_pretrained_model is not None
                    class_rewards['class_eval'][step_t] = \
                        parallel_worker.test_fixed_classifier(global_t=global_t,
                                                    max_steps=args.eval_max_steps,
                                                    sess=pretrain_sess,
                                                    worker=all_workers[-1],
                                                    model=global_pretrained_model)
                    class_last_reward = class_rewards['class_eval'][step_t][0]

                checkpt_file = folder / 'model_checkpoints'
                checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
                saver.save(sess, str(checkpt_file), global_step=global_t)
                save_best_model(rewards['eval'][global_t][0])

                # saving worker info to dicts
                # SIL
                sil_dict['sil_ctr'][step_t] = sil_ctr
                sil_dict['sil_a3c_sampled'][step_t] = sil_a3c_sampled
                sil_dict['sil_a3c_used'][step_t] = sil_a3c_used
                sil_dict['sil_a3c_sampled_return'][step_t] = sil_a3c_sampled_return
                sil_dict['sil_a3c_used_return'][step_t] = sil_a3c_used_return
                sil_dict['sil_a3c_used_adv'][step_t] = sil_a3c_used_adv
                sil_dict['sil_rollout_sampled'][step_t] = sil_rollout_sampled
                sil_dict['sil_rollout_used'][step_t] = sil_rollout_used
                sil_dict['sil_rollout_sampled_return'][step_t] = sil_rollout_sampled_return
                sil_dict['sil_rollout_used_return'][step_t] = sil_rollout_used_return
                sil_dict['sil_rollout_used_adv'][step_t] = sil_rollout_used_adv
                sil_dict['sil_old_sampled'][step_t] = sil_old_sampled
                sil_dict['sil_old_used'][step_t] = sil_old_used
                # ROLLOUT
                rollout_dict['rollout_ctr'][step_t] = rollout_ctr
                rollout_dict['rollout_added_ctr'][step_t] = rollout_added_ctr
                rollout_dict['rollout_steps'][step_t] = rollout_steps
                rollout_dict['rollout_sample_used'][step_t] = rollout_sample_used
                rollout_dict['rollout_sample_used_adv'][step_t] = rollout_sample_used_adv
                rollout_dict['rollout_new_return'][step_t] = rollout_new_return
                rollout_dict['rollout_old_return'][step_t] = rollout_old_return
                # A3C
                a3c_dict['a3c_ctr'][step_t] = a3c_ctr
                a3c_dict['a3c_sample_used'][step_t] = a3c_sample_used
                a3c_dict['a3c_sample_used_return'][step_t] = a3c_sample_used_return
                a3c_dict['a3c_sample_used_adv'][step_t] = a3c_sample_used_adv

                # dump pickle
                # not saving class_reward
                dump_pickle([rewards, sil_dict, rollout_dict, a3c_dict],
                            [reward_fname, sil_fname, rollout_fname, a3c_fname])
                if args.load_pretrained_model:
                    dump_pickle([class_rewards], [class_reward_fname])

                logger.info('Dump pickle at step {}'.format(global_t))

                # save replay buffer (only works for priority mem for now)
                if args.checkpoint_buffer:
                    if shared_memory is not None and args.priority_memory:
                        params = [shared_memory.buff._next_idx, shared_memory.buff._max_priority]
                        trees = [shared_memory.buff._it_sum._value,
                                 shared_memory.buff._it_min._value]
                        dump_pickle([shared_memory.buff._storage, params, trees],
                                    [sharedmem_fname, sharedmem_params_fname, sharedmem_trees_fname])
                        logger.info('Saving shared_memory')

                    if rollout_buffer is not None and args.priority_memory:
                        params = [rollout_buffer.buff._next_idx, rollout_buffer.buff._max_priority]
                        trees = [rollout_buffer.buff._it_sum._value,
                                 rollout_buffer.buff._it_min._value]
                        dump_pickle([rollout_buffer.buff._storage, params, trees],
                                    [rolloutmem_fname, rolloutmem_params_fname, rolloutmem_trees_fname])
                        logger.info('Saving rollout_buffer')

                step_t = 1

        # set start_time
        start_time = time.time() - wall_t
        parallel_worker.set_start_time(start_time)

        if parallel_worker.is_sil_thread:
            # TODO(add as command-line parameters later)
            # next_idx = 0
            sil_interval = 0  # bigger number => slower SIL updates
            m_repeat = 4
            min_mem = args.batch_size * m_repeat
            sil_train_flag = args.load_memory and len(shared_memory) >= min_mem

        elif parallel_worker.is_rollout_thread:
            pass

        elif parallel_worker.is_classify_thread:
            # TODO(add as command-line parameters later)
            classify_ctr = 0
            classify_interval = 100  # bigger number => slower classification updates
            classify_train_steps = args.class_train_steps
            # re-train if we have accumulated enough new samples
            # also consider the training time (about 6 min for 20k steps on GTX1060)
            classify_min_mem = args.batch_size * classify_train_steps * 0.1
            classify_train_flag = args.load_memory and \
                len(exp_buffer) >= classify_min_mem

        while True:
            if stop_req:
                return

            if global_t >= (args.max_time_step * args.max_time_step_fraction):
                return

            if parallel_worker.is_sil_thread:
                if args.use_sil_skip:
                    if net_updates.qsize() >= sil_interval \
                       and len(shared_memory) >= min_mem \
                       and global_t % 2 == 0:
                        sil_train_flag = True
                else:
                    if net_updates.qsize() >= sil_interval \
                       and len(shared_memory) >= min_mem:
                        sil_train_flag = True

                if sil_train_flag:
                    sil_train_flag = False
                    th_ctr.get()
                    # when multiSIL, all SIL threads need to be done with training before adding ep
                    # otherwise, mess up with priority update in shared_memory
                    th_sil_ctr.get()

                    if args.stop_rollout and class_last_reward <= last_reward:
                        # rollout_sample_proportion = 0
                        _stop_rollout = True

                    train_out = parallel_worker.sil_train(
                        sess, global_t, shared_memory, m_repeat,
                        # sil_ctr,
                        # sil_a3c_sampled, sil_a3c_used,
                        # sil_a3c_sampled_return, sil_a3c_used_return,
                        # sil_a3c_used_adv,
                        # sil_rollout_sampled, sil_rollout_used,
                        # sil_rollout_sampled_return, sil_rollout_used_return,
                        # sil_rollout_used_adv,
                        # sil_old_sampled, sil_old_used,
                        rollout_buffer=rollout_buffer,
                        # rollout_sample_proportion=rollout_sample_proportion,
                        stop_rollout=_stop_rollout,
                        roll_any=args.roll_any)

                    local_sil_ctr, local_sil_a3c_sampled, local_sil_a3c_used, \
                    local_sil_a3c_sampled_return, local_sil_a3c_used_return, local_sil_a3c_used_adv,\
                    local_sil_rollout_sampled, local_sil_rollout_used, \
                    local_sil_rollout_sampled_return, local_sil_rollout_used_return, local_sil_rollout_used_adv,\
                    local_sil_old_sampled, local_sil_old_used, goodstate, badstate = train_out

                    th_sil_ctr.put(1)
                    th_ctr.put(1)

                    with net_updates.mutex:
                        net_updates.queue.clear()

                    if args.train_classifier:
                        class_updates.put(1)
                        while not goodstate.empty():
                            goodstate_queue.put(goodstate.get())

                    # add to badstate_queue if we are rolling out specifically from bad states only
                    # otherwise, just let rollout sample from shared_memory
                    if args.use_rollout and not args.roll_any:
                        while not badstate.empty():
                            if badstate_queue.full():
                                #TODO we are removing the ones with smallest adv
                                # should we remove the largest adv first?
                                # that will be a reversed priority queue
                                # but for now as it is
                                badstate_queue.get()
                            badstate_queue.put(badstate.get())

                    with goodstate.mutex:
                        goodstate.queue.clear()
                    with badstate.mutex:
                        badstate.queue.clear()

                    if args.use_rollout or args.train_classifier:
                        # parallel_worker.record_sil(sil_ctr=sil_ctr,
                        #                       total_used=(sil_a3c_used + sil_rollout_used),
                        #                       num_a3c_used=sil_a3c_used,
                        #                       a3c_used_return=sil_a3c_used_return/(sil_a3c_used+1),
                        #                       rollout_sampled=sil_rollout_sampled,
                        #                       rollout_used=sil_rollout_used,
                        #                       rollout_used_return=sil_rollout_used_return/(sil_rollout_used+1),
                        #                       old_sampled=sil_old_sampled,
                        #                       old_used=sil_old_used,
                        #                       goods=goodstate_queue.qsize(),
                        #                       bads=badstate_queue.qsize(),
                        #                       global_t=global_t)

                        if sil_ctr % (args.num_sil_worker*100) <= 3 and sil_ctr > 0:
                            rollout_buffsize = 0
                            if not args.one_buffer:
                                rollout_buffsize = len(rollout_buffer)
                            log_data = (sil_ctr, len(shared_memory),
                                        rollout_buffsize,
                                        sil_a3c_used+sil_rollout_used,
                                        args.batch_size*sil_ctr,
                                        sil_a3c_used,
                                        sil_a3c_used_return/(sil_a3c_used+1),
                                        sil_a3c_used_adv/(sil_a3c_used+1),
                                        sil_rollout_sampled,
                                        sil_rollout_used,
                                        sil_rollout_used_return/(sil_rollout_used+1),
                                        sil_rollout_used_adv/(sil_rollout_used+1),
                                        sil_old_sampled,
                                        sil_old_used,
                                        goodstate_queue.qsize(),
                                        badstate_queue.qsize())
                            logger.info("SIL: sil_ctr={0:}"
                                        " sil_memory_size={1:}"
                                        " rollout_buffer_size={2:}"
                                        " total_sample_used={3:}/{4:}"
                                        " a3c_used={5:}"
                                        " a3c_used_return_avg={6:.2f}"
                                        " a3c_used_adv_avg={7:.2f}"
                                        " rollout_sampled={8:}"
                                        " rollout_used={9:}"
                                        " rollout_used_return_avg={10:.2f}"
                                        " rollout_used_adv_avg={11:.2f}"
                                        " old_sampled={12:}"
                                        " old_used={13:}"
                                        " #good_states={14:}"
                                        " #bad_states={15:}".format(*log_data))
                    else:
                        # parallel_worker.record_sil(sil_ctr=sil_ctr,
                        #                            total_used=(sil_a3c_used + sil_rollout_used),
                        #                            num_a3c_used=sil_a3c_used,
                        #                            rollout_sampled=sil_rollout_sampled,
                        #                            rollout_used=sil_rollout_used,
                        #                            global_t=global_t)
                        if sil_ctr % (args.num_sil_worker*100) <= 3 and sil_ctr > 0:
                            log_data = (sil_ctr, sil_a3c_used+sil_rollout_used,
                                        args.batch_size*sil_ctr,
                                        sil_a3c_used,
                                        len(shared_memory))
                            logger.info("SIL: sil_ctr={0:}"
                                        " total_sample_used={1:}/{2:}"
                                        " a3c_used={3:}"
                                        " sil_memory_size={4:}".format(*log_data))
                else: # before sil starts, init local count
                    local_sil_ctr, local_sil_a3c_sampled, local_sil_a3c_used = 0, 0, 0
                    local_sil_a3c_sampled_return, local_sil_a3c_used_return = 0, 0
                    local_sil_a3c_used_adv, local_sil_rollout_sampled, local_sil_rollout_used = 0, 0, 0
                    local_sil_rollout_sampled_return, local_sil_rollout_used_return = 0, 0
                    local_sil_rollout_used_adv, local_sil_old_sampled, local_sil_old_used = 0, 0, 0

                # Adding episodes to SIL memory is centralize to ensure
                # sampling and updating of priorities does not become a problem
                # since we add new episodes to SIL at once and during
                # SIL training it is guaranteed that SIL memory is untouched.
                # wait for #a3c worker episodes, so that we don't need to pause at every step
                # but we don't want to wait for too long either
                # with more SIL workers, waiting too long could result in SIL updating with too much old mem
                max = args.parallel_size - args.num_sil_worker # + args.num_rollout_worker)
                q_size = ep_queue.qsize()
                if args.num_sil_worker > 1: # when milti SIL, make sure only one is adding at a time
                    if q_size >= max:
                        with sil_lock:
                            # only allow one thread to add to shared_memory
                            if parallel_worker.thread_idx == sil_addmem_thread:
                                # wait for every thread done training
                                while th_sil_ctr.qsize() < args.num_sil_worker:
                                    time.sleep(0.001)
                                while not ep_queue.empty():
                                    data = ep_queue.get()
                                    parallel_worker.episode.set_data(*data)
                                    shared_memory.extend(parallel_worker.episode)
                                    parallel_worker.episode.reset()
                                    q_size -= 1
                                    # This ensures SIL doesn't addmem non-stop and has a chance to train
                                    if q_size <= 0:
                                        break
                else: # single SIL doesn't need to wait
                    while not ep_queue.empty():
                        data = ep_queue.get()
                        parallel_worker.episode.set_data(*data)
                        shared_memory.extend(parallel_worker.episode)
                        parallel_worker.episode.reset()
                        max -= 1
                        # This ensures that SIL has a chance to train
                        if max <= 0:
                            break

                # # at sil_interval=0, this will never be executed,
                # # this is considered fast SIL (no intervals between updates)
                # if not sil_train_flag and len(shared_memory) >= min_mem:
                #     # No training, just updating priorities
                #     parallel_worker.sil_update_priorities(
                #         sess, shared_memory, m=m_repeat)

                diff_global_t = 0
                # for centralized A3C counting
                local_a3c_ctr = 0
                local_a3c_sample_used = 0
                local_a3c_sample_used_return = 0
                local_a3c_sample_used_adv = 0


            elif parallel_worker.is_rollout_thread:
                th_ctr.get()
                diff_global_t = 0

                if global_t < args.advice_budget and \
                   global_t > (args.delay_rollout*args.eval_freq) and \
                   len(shared_memory) >= 1: # rollout_sample_proportion > 0 and \

                    if not args.roll_any:
                        _, _, sample = badstate_queue.get()
                    else:
                        sample = shared_memory.sample_one_random()
                        # after sample, refreshed should be True
                        assert sample[-1] == True

                    train_out = parallel_worker.rollout(sess, folder, pretrain_sess,
                                                   GAME_NAME, global_t, sample,
                                                   rollout_ctr, rollout_added_ctr,
                                                   rollout_sample_used,
                                                   rollout_sample_used_adv,
                                                   rollout_new_return,
                                                   rollout_old_return,
                                                   args.add_all_rollout,
                                                   args.max_ep_step,
                                                   args.nstep_bc,
                                                   args.update_in_rollout)

                    diff_global_t, episode_end, part_end, rollout_ctr, \
                        rollout_added_ctr, add, rollout_sample_used, rollout_sample_used_adv, \
                        rollout_new_return, rollout_old_return= train_out

                    rollout_steps += diff_global_t

                    if rollout_ctr % 10 == 0 and rollout_ctr > 0:
                        logger.info("ROLLOUT total steps: {}".format(rollout_steps))
                        logger.info("A3C total steps: {}".format(global_t - rollout_steps))

                    # should always part_end,
                    # and only add if new return is better
                    if part_end and add:
                        if not args.one_buffer:
                            rollout_buffer.extend(parallel_worker.episode)
                        else: # 10/16 fix: should not mess with SIL's index, just add to ep_queue
                            # shared_memory.extend(parallel_worker.episode)
                            ep_queue.put(parallel_worker.episode.get_data())

                    parallel_worker.episode.reset()

                # for centralized A3C counting
                local_a3c_ctr = 0
                local_a3c_sample_used = 0
                local_a3c_sample_used_return = 0
                local_a3c_sample_used_adv = 0

                local_sil_ctr, local_sil_a3c_sampled, local_sil_a3c_used = 0, 0, 0
                local_sil_a3c_sampled_return, local_sil_a3c_used_return = 0, 0
                local_sil_a3c_used_adv, local_sil_rollout_sampled, local_sil_rollout_used = 0, 0, 0
                local_sil_rollout_sampled_return, local_sil_rollout_used_return = 0, 0
                local_sil_rollout_used_adv, local_sil_old_sampled, local_sil_old_used = 0, 0, 0

                th_ctr.put(1)

            elif parallel_worker.is_classify_thread:
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
                        s, fs, a, _return, _rollout = data
                        exp_buffer.extend_one(s, fs, a, _return, _rollout)
                        count -= 1
                        if count <= 0:
                            break

                if classify_train_flag:
                    classify_train_flag = False
                    logger.info("Start training classifier...")
                    th_ctr.get()

                    classify_ctr = parallel_worker.train(
                        pretrain_sess, global_t, exp_buffer, classify_ctr)

                    th_ctr.put(1)
                    logger.info("Finish training classifier")

                    with class_updates.mutex:
                        class_updates.queue.clear()

                diff_global_t = 0
                # for centralized A3C counting
                local_a3c_ctr = 0
                local_a3c_sample_used = 0

            # a3c training thread worker
            else:
                th_ctr.get()

                train_out = parallel_worker.train(sess, global_t, rewards)
                diff_global_t, episode_end, part_end, \
                    local_a3c_ctr, local_a3c_sample_used, \
                    local_a3c_sample_used_return, local_a3c_sample_used_adv = train_out

                th_ctr.put(1)

                if args.use_sil:
                    net_updates.put(1)
                    if part_end:
                        ep_queue.put(parallel_worker.episode.get_data())
                        parallel_worker.episode.reset()

                local_sil_ctr, local_sil_a3c_sampled, local_sil_a3c_used = 0, 0, 0
                local_sil_a3c_sampled_return, local_sil_a3c_used_return = 0, 0
                local_sil_a3c_used_adv, local_sil_rollout_sampled, local_sil_rollout_used = 0, 0, 0
                local_sil_rollout_sampled_return, local_sil_rollout_used_return = 0, 0
                local_sil_rollout_used_adv, local_sil_old_sampled, local_sil_old_used = 0, 0, 0

            # ensure only one thread is updating global_t at a time
            with lock:
                global_t += diff_global_t

                # centralize increasing count for A3C , SIL and Rollout
                a3c_ctr += local_a3c_ctr
                a3c_sample_used += local_a3c_sample_used
                a3c_sample_used_return += local_a3c_sample_used_return
                a3c_sample_used_adv += local_a3c_sample_used_adv

                sil_ctr += local_sil_ctr
                sil_a3c_sampled += local_sil_a3c_sampled
                sil_a3c_used += local_sil_a3c_used
                sil_a3c_sampled_return += local_sil_a3c_sampled_return
                sil_a3c_used_return += local_sil_a3c_used_return
                sil_a3c_used_adv += local_sil_a3c_used_adv
                sil_rollout_sampled += local_sil_rollout_sampled
                sil_rollout_used += local_sil_rollout_used
                sil_rollout_sampled_return += local_sil_rollout_sampled_return
                sil_rollout_used_return += local_sil_rollout_used_return
                sil_rollout_used_adv += local_sil_rollout_used_adv
                sil_old_sampled += local_sil_old_sampled
                sil_old_used += local_sil_old_used

                # if during a thread's update, global_t has reached a evaluation interval
                if global_t > next_global_t:

                    logger.info("A3C episode return avg: {:.2f}".format(a3c_sample_used_return/a3c_sample_used))
                    logger.info("A3C episode adv avg: {:.2f}".format(a3c_sample_used_adv/a3c_sample_used))

                    next_global_t = next_t(global_t, args.eval_freq)
                    step_t = int(next_global_t - args.eval_freq)

                    # wait 1 hour max for all threads to finish before testing
                    time_waited = 0
                    while not stop_req and th_ctr.qsize() < len(all_workers):
                        time.sleep(0.001)
                        time_waited += 0.001
                        if time_waited > 3600:
                            sys.exit('Exceed waiting time, exiting...')

                    step_t = int(next_global_t - args.eval_freq)

                    rewards['eval'][step_t] = parallel_worker.testing(
                        sess, args.eval_max_steps, step_t, folder,
                        demo_memory_cam=demo_memory_cam, worker=all_workers[-1])
                    save_best_model(rewards['eval'][step_t][0])
                    last_reward = rewards['eval'][step_t][0]

                    # saving worker info to dicts
                    # SIL
                    sil_dict['sil_ctr'][step_t] = sil_ctr
                    sil_dict['sil_a3c_sampled'][step_t] = sil_a3c_sampled
                    sil_dict['sil_a3c_used'][step_t] = sil_a3c_used
                    sil_dict['sil_a3c_sampled_return'][step_t] = sil_a3c_sampled_return
                    sil_dict['sil_a3c_used_return'][step_t] = sil_a3c_used_return
                    sil_dict['sil_a3c_used_adv'][step_t] = sil_a3c_used_adv
                    sil_dict['sil_rollout_sampled'][step_t] = sil_rollout_sampled
                    sil_dict['sil_rollout_used'][step_t] = sil_rollout_used
                    sil_dict['sil_rollout_sampled_return'][step_t] = sil_rollout_sampled_return
                    sil_dict['sil_rollout_used_return'][step_t] = sil_rollout_used_return
                    sil_dict['sil_rollout_used_adv'][step_t] = sil_rollout_used_adv
                    sil_dict['sil_old_sampled'][step_t] = sil_old_sampled
                    sil_dict['sil_old_used'][step_t] = sil_old_used
                    # ROLLOUT
                    rollout_dict['rollout_ctr'][step_t] = rollout_ctr
                    rollout_dict['rollout_added_ctr'][step_t] = rollout_added_ctr
                    rollout_dict['rollout_steps'][step_t] = rollout_steps
                    rollout_dict['rollout_sample_used'][step_t] = rollout_sample_used
                    rollout_dict['rollout_sample_used_adv'][step_t] = rollout_sample_used_adv
                    rollout_dict['rollout_new_return'][step_t] = rollout_new_return
                    rollout_dict['rollout_old_return'][step_t] = rollout_old_return
                    # A3C
                    a3c_dict['a3c_ctr'][step_t] = a3c_ctr
                    a3c_dict['a3c_sample_used'][step_t] = a3c_sample_used
                    a3c_dict['a3c_sample_used_return'][step_t] = a3c_sample_used_return
                    a3c_dict['a3c_sample_used_adv'][step_t] = a3c_sample_used_adv

                    # testing classifier in game (retrain classifier)
                    if args.train_classifier:
                        assert classifier_index is not None
                        class_rewards['class_eval'][step_t] = \
                            parallel_worker.test_retrain_classifier(global_t=step_t,
                                                        max_steps=args.eval_max_steps,
                                                        sess=pretrain_sess,
                                                        worker=all_workers[classifier_index])
                        class_last_reward = class_rewards['class_eval'][step_t][0]
                    # testing classifier in game (fix classifier)
                    elif args.load_pretrained_model:
                        assert pretrain_sess is not None
                        assert global_pretrained_model is not None
                        class_rewards['class_eval'][step_t] = \
                            parallel_worker.test_fixed_classifier(global_t=step_t,
                                                        max_steps=args.eval_max_steps,
                                                        sess=pretrain_sess,
                                                        worker=all_workers[-1],
                                                        model=global_pretrained_model)
                        class_last_reward = class_rewards['class_eval'][step_t][0]

                if global_t > next_save_t:
                    # freq = (args.max_time_step * args.max_time_step_fraction)
                    # freq = freq // 5
                    next_save_t = next_t(global_t, args.eval_freq*args.checkpoint_freq)#freq)
                    # save a3c
                    checkpt_file = folder / 'model_checkpoints'
                    checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
                    saver.save(sess, str(checkpt_file), global_step=global_t,
                            write_meta_graph=False)

                    # dump pickle
                    dump_pickle([rewards, sil_dict, rollout_dict, a3c_dict],
                                [reward_fname, sil_fname, rollout_fname, a3c_fname])
                    if args.load_pretrained_model:
                        dump_pickle([class_rewards], [class_reward_fname])

                    logger.info('Dump pickle at step {}'.format(global_t))

                    # save replay buffer (only works for priority mem for now)
                    if args.checkpoint_buffer:
                        if shared_memory is not None and args.priority_memory:
                            params = [shared_memory.buff._next_idx, shared_memory.buff._max_priority]
                            trees = [shared_memory.buff._it_sum._value,
                                     shared_memory.buff._it_min._value]
                            dump_pickle([shared_memory.buff._storage, params, trees],
                                        [sharedmem_fname, sharedmem_params_fname, sharedmem_trees_fname])
                            logger.info('Saving shared_memory')

                        if rollout_buffer is not None and args.priority_memory:
                            params = [rollout_buffer.buff._next_idx, rollout_buffer.buff._max_priority]
                            trees = [rollout_buffer.buff._it_sum._value,
                                     rollout_buffer.buff._it_min._value]
                            dump_pickle([rollout_buffer.buff._storage, params, trees],
                                        [rolloutmem_fname, rolloutmem_params_fname, rolloutmem_trees_fname])
                            logger.info('Saving rollout_buffer')


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


    train_threads = []
    th_ctr = Queue()
    for i in range(args.parallel_size):
        th_ctr.put(1)
    episodes_queue = None
    net_updates = None
    class_updates = None
    goodstate_queue = None
    badstate_queue = None
    badstate_list = []
    th_sil_ctr = None
    if args.use_sil:
        episodes_queue = Queue()
        net_updates = Queue()
        th_sil_ctr = Queue()
        for i in range(args.num_sil_worker):
            th_sil_ctr.put(1)
    if args.train_classifier or args.use_rollout:
        class_updates = Queue()
        goodstate_queue = Queue()
        badstate_queue = PriorityQueue(maxsize=args.memory_length)
    for i in range(args.parallel_size):
        worker_thread = Thread(
            target=train_function,
            args=(i, th_ctr, episodes_queue, net_updates, class_updates,
                goodstate_queue, badstate_queue, badstate_list,th_sil_ctr,))
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

    dump_pickle([rewards, sil_dict, rollout_dict, a3c_dict],
                [reward_fname, sil_fname, rollout_fname, a3c_fname])

    logger.info('Data saved!')

    sess.close()
    if pretrain_sess:
        pretrain_sess.close()
