#!/usr/bin/env python3
"""Autoencoder as pre-training method.

This module uses Autoencoder as a pre-training method to learn features
using data from human demonstrations. It has options to further use features
as starting parameters for classification. Another option is to use
Supervised Autoencoder (SAE) that jointly trains the Autoencoder and
classification of human demonstration data.

Example:
    $ python3 pretrain/run_experiment.py
        --gym-env=PongNoFrameskip-v4
        --ae-classify-demo --use-mnih-2015
        --train-max-steps=150000 --batch_size=32
"""
import cv2
import logging
import numpy as np
import os
import pathlib
import signal
import sys

from common.game_state import GameState
from common.game_state import get_wrapper_by_name
from common.replay_memory import ReplayMemoryReturns
from common.util import load_memory
from common.util import LogFormatter
from common.util import percent_decrease
from class_network import AutoEncoderNetwork
from termcolor import colored

logger = logging.getLogger("ae_classify_demo")


class AutoencoderClassifyDemo(object):
    """Use Autoencoder for learning features."""

    def __init__(self, tf, net, net_sess, name, train_max_steps, batch_size,
                 ae_grad_applier, grad_applier, eval_freq=5000,
                 demo_memory_folder=None, demo_ids=None, folder=None,
                 exclude_num_demo_ep=0, device='/cpu:0', clip_norm=None, game_state=None,
                 sampling_type=None, sl_loss_weight=1.0, reward_constant=0,
                 classify_thread=False, sil_thread=False):
        """Initialize AutoencoderClassifyDemo class."""
        assert game_state is not None

        self.net = net
        self.net_sess = net_sess
        self.name = name
        self.train_max_steps = train_max_steps
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.folder = folder
        self.tf = tf
        self.exclude_num_demo_ep = exclude_num_demo_ep
        self.stop_requested = False
        self.game_state = game_state
        self.best_model_reward = -(sys.maxsize)
        self.sampling_type = sampling_type
        self.classify_thread = classify_thread
        self.sil_thread = False

        logger.info("Classfier:")
        logger.info("train_max_steps: {}".format(self.train_max_steps))
        logger.info("batch_size: {}".format(self.batch_size))
        logger.info("eval_freq: {}".format(self.eval_freq))
        logger.info("sampling_type: {}".format(self.sampling_type))
        logger.info("sl_loss_weight: {}".format(sl_loss_weight))
        logger.info("clip_norm: {}".format(clip_norm))
        logger.info("reward_constant: {}".format(reward_constant))
        logger.info("device: {}".format(device))
        logger.info("classifier_thread: {}".format(
            colored(self.classify_thread, "green" if self.classify_thread else "red")))
        logger.info("sil_thread: {}".format(
            colored(self.sil_thread, "green" if self.sil_thread else "red")))


        # self.demo_memory, acts_ctr, total_rewards, total_steps = \
        #     load_memory(name=None, demo_memory_folder=demo_memory_folder,
        #                 demo_ids=demo_ids)

        # self.action_dist = [acts_ctr[a] for a in range(self.net.action_size)]

        # self.net.prepare_loss(sl_loss_weight=sl_loss_weight,
        #                       val_weight=0.5 * sl_loss_weight)
        self.net.prepare_loss(sl_loss_weight=1.0, val_weight=0.005)
        self.net.prepare_evaluate()
        self.apply_gradients = self.prepare_compute_gradients(
            grad_applier, device, clip_norm=clip_norm)

        # Autoencoder optimizer
        with self.net.graph.as_default():
            with self.tf.device(device):
                self.ae_apply_gradients = None
                if not self.net.sae:
                    self.ae_apply_gradients = (ae_grad_applier
                                               .minimize(self.net.ae_loss))

    def prepare_compute_gradients(self, grad_applier, device, clip_norm=None):
        """Return operation for gradient application.

        Keyword arguments:
        grad_applier -- optimizer for applying gradients
        device -- cpu or gpu
        clip_norm -- value for clip_by_global_norm (default None)
        """
        with self.net.graph.as_default():
            with self.tf.device(device):
                apply_gradients = self.__compute_gradients(
                    grad_applier, self.net.total_loss, clip_norm)

        return apply_gradients

    def __compute_gradients(self, grad_applier, total_loss, clip_norm=None):
        """Apply gradient clipping and return op for gradient application."""
        grads_vars = grad_applier.compute_gradients(
            total_loss, self.net.get_vars())
        grads = []
        params = []

        for p in grads_vars:
            if p[0] is None:
                continue
            grads.append(p[0])
            params.append(p[1])

        if clip_norm is not None:
            grads, _ = self.tf.clip_by_global_norm(grads, clip_norm)

        grads_vars_updates = zip(grads, params)
        return grad_applier.apply_gradients(grads_vars_updates)

    def save_best_model(self, test_reward, best_saver, sess):
        """Save best network model's parameters and reward to file.

        Keyword arguments:
        test_reward -- testing total average reward
        best_saver -- tf saver object
        sess -- tf session
        """
        self.best_model_reward = test_reward
        best_model_reward_file = self.folder / 'model_best/best_model_reward'
        with best_model_reward_file.open('w') as f:
            f.write(str(self.best_model_reward))

        best_file = self.folder / 'model_best'
        best_file /= '{}_checkpoint'.format(self.name.replace('-', '_'))
        best_saver.save(sess, str(best_file))

    def set_summary_writer(self, writer):
        """Set summary writer."""
        self.writer = writer

    def record_summary(self, score=0, steps=0, episodes=None, global_t=0,
                       mode='Class_Test'):
        """Record summary."""
        summary = self.tf.Summary()
        summary.value.add(tag='{}/score'.format(mode),
                          simple_value=float(score))
        summary.value.add(tag='{}/steps'.format(mode),
                          simple_value=float(steps))
        if episodes is not None:
            summary.value.add(tag='{}/episodes'.format(mode),
                              simple_value=float(episodes))
        self.writer.add_summary(summary, global_t)
        self.writer.flush()

    def choose_action_with_high_confidence(self, pi_values, exclude_noop=True):
        """Return action with highest confidence.

        Keyword arguments:
        pi_values -- neural network softmax output pi_values
        exclude_noop -- exclude no-operation action (default True)
        """
        max_confidence_action = np.argmax(pi_values[1 if exclude_noop else 0:])
        confidence = pi_values[max_confidence_action]
        return (max_confidence_action+(1 if exclude_noop else 0)), confidence

    def test_game(self, sess, max_steps, global_t):
        """Evaluate game with current network model.

        Keyword argument:
        sess -- tf session
        """
        self.game_state.reset(hard_reset=True)

        # max_steps = 25000
        total_reward = 0
        total_steps = 0
        episode_reward = 0
        episode_steps = 0
        n_episodes = 0
        while max_steps > 0:
            state = cv2.resize(self.game_state.s_t,
                               self.net.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)
            model_pi = self.net.run_policy(sess, state)
            action, confidence = self.choose_action_with_high_confidence(
                model_pi, exclude_noop=False)

            # take action
            self.game_state.step(action)
            terminal = self.game_state.terminal
            episode_reward += self.game_state.reward
            episode_steps += 1
            max_steps -= 1

            # s_t = s_t1
            self.game_state.update()

            if terminal:
                was_real_done = get_wrapper_by_name(
                    self.game_state.env, 'EpisodicLifeEnv').was_real_done

                if was_real_done:
                    n_episodes += 1
                    score_str = colored("score={}".format(
                        episode_reward), "magenta")
                    steps_str = colored("steps={}".format(
                        episode_steps), "blue")
                    log_data = (n_episodes, score_str, steps_str, total_steps)
                    logger.debug("test: trial={} {} {} total_steps={}"
                                 .format(*log_data))
                    total_reward += episode_reward
                    total_steps += episode_steps
                    episode_reward = 0
                    episode_steps = 0

                self.game_state.reset(hard_reset=False)

        if n_episodes == 0:
            total_reward = episode_reward
            total_steps = episode_steps
        else:
            total_reward = total_reward / n_episodes
            total_steps = total_steps // n_episodes

        log_data = (global_t, total_reward, total_steps, n_episodes)
        logger.info("test global_t={}: final score={} final steps={} # trials={}"
                    .format(*log_data))
        self.record_summary(
            score=total_reward, steps=total_steps,
            episodes=n_episodes, global_t=global_t, mode='Class_Test')
        return (total_reward, total_steps, n_episodes)

    def train_autoencoder(self, sess, summary_op, summary_writer):
        """Train Autoencoder with human demonstration.

        Keyword arguments:
        sess -- tf session
        summary_op -- tf summary operation
        summary_writer -- tf summary writer
        """
        prev_ae_loss = 0

        for i in range(self.train_max_steps + 1):
            if self.stop_requested:
                break

            batch_si, _, _, _, batch_returns = \
                self.combined_memory.sample_nowrap(self.batch_size,
                                                   self.action_dist,
                                                   type=self.sampling_type)

            feed_dict = {self.net.s: batch_si}

            if self.net.use_denoising:
                feed_dict[self.net.training] = True
            if self.net.use_slv:
                feed_dict[self.net.returns] = batch_returns

            ae_loss, _ = sess.run(
                [self.net.ae_loss, self.ae_apply_gradients],
                feed_dict=feed_dict)

            ae_percent_change = percent_decrease(ae_loss, prev_ae_loss)
            if i % self.eval_freq == 0 or ae_percent_change > 2.5:
                prev_ae_loss = ae_loss
                logger.debug("i={0:} loss={1:.6f}".format(i, ae_loss))

                summary = self.tf.Summary()
                summary.value.add(tag='AE_Train_Loss',
                                  simple_value=float(ae_loss))

                summary_writer.add_summary(summary, i)
                summary_writer.flush()

                first_image = (self.combined_memory[0])[0]
                first_stack = np.vstack((
                    first_image[:, :, 0], first_image[:, :, 1],
                    first_image[:, :, 2], first_image[:, :, 3]))
                out_image = self.net.reconstruct_image(sess, first_image)
                out_image = np.uint8(out_image * 255)
                out_stack = np.vstack((
                    out_image[:, :, 0], out_image[:, :, 1],
                    out_image[:, :, 2], out_image[:, :, 3]))
                side_by_side = np.hstack((first_stack, out_stack))
                file = self.folder / "ae_{0:07d}.png".format(i)
                cv2.imwrite(str(file), side_by_side)

    def train(self, sess, exp_buffer, classify_ctr):
        #, summary_op, summary_writer, best_saver=None):
        """Train classification with human demonstration.

        This method either does classification training after autoencoder
        training or jointly trains autoencoder and classification using
        Supervised Autoencoder (SAE).

        Keyword arguments:
        sess -- tf session
        exp_buffer -- experience buffer Q to sample from
        classify_ctr -- count number of retrain on classifier
        """
        self.max_val = -(sys.maxsize)
        prev_ae_loss = 0
        train_max_steps = self.train_max_steps
        # if not self.net.sae:
        #     train_max_steps = 50000

        for i in range(train_max_steps + 1):
            if self.stop_requested:
                break

            #TODO: oversample
            sample = exp_buffer.sample(self.batch_size, beta=0.4)
            index_list, batch, weights = sample
            batch_si, batch_a, batch_returns, batch_fullstate = batch

            feed_dict = {self.net.s: batch_si, self.net.a: batch_a}

            if self.net.use_denoising:
                feed_dict[self.net.training] = True
            if self.net.use_slv:
                feed_dict[self.net.returns] = batch_returns

            if self.net.sae:  # supervised autoencoder
                out = sess.run([
                    self.net.total_loss,
                    self.net.sl_loss,
                    self.net.ae_loss,
                    self.net.accuracy,
                    self.net.max_value,
                    self.apply_gradients],
                    feed_dict=feed_dict)
                train_loss, sl_loss, ae_loss, acc, max_value, _ = out
            else:
                ae_loss = 0
                out = sess.run(
                    [self.net.total_loss,
                     self.net.sl_loss,
                     self.net.accuracy,
                     self.net.max_value,
                     self.apply_gradients],
                    feed_dict=feed_dict)
                train_loss, sl_loss, acc, max_value, _ = out

            if max_value > self.max_val:
                self.max_val = max_value

            # ae_percent_change = percent_decrease(ae_loss, prev_ae_loss)
            # if i % self.eval_freq == 0 or ae_percent_change > 2.5:
            #     prev_ae_loss = ae_loss
            #
            #     test_acc = sess.run(
            #         self.net.accuracy,
            #         feed_dict={
            #             self.net.s: self.test_batch_si,
            #             self.net.a: self.test_batch_a})
            #
            #     summary = self.tf.Summary()
            #     summary.value.add(tag='SL_Loss', simple_value=float(sl_loss))
            #     summary.value.add(tag='Train_Loss',
            #                       simple_value=float(train_loss))
            #     summary.value.add(tag='Accuracy', simple_value=float(test_acc))
            #
            #     if self.net.sae:
            #         summary.value.add(tag='AE_Train_Loss',
            #                           simple_value=float(ae_loss))
            #         logger.debug("i={0:} train_acc={1:.4f} test_acc={2:.4f}"
            #                      " loss={3:.4f} sl_loss={4:.8f}"
            #                      " ae_loss={5:.6f} max_val={6:.4f}".format(
            #                       i, acc, test_acc, train_loss, sl_loss,
            #                       ae_loss, self.max_val))
            #     else:
            #         logger.debug("i={0:} train_acc={1:.4f} test_acc={2:.4f}"
            #                      " loss={3:.8f} max_val={4:.4f}".format(
            #                       i, acc, test_acc, train_loss, self.max_val))
            #
            #     self.writer.add_summary(summary, i)
            #     self.writer.flush()
            #
            #     # if self.net.sae and i % (self.train_max_steps // 15) == 0:
            #     if self.net.sae:
            #         first_image = (self.combined_memory[0])[0]
            #         first_stack = np.vstack((
            #             first_image[:, :, 0], first_image[:, :, 1],
            #             first_image[:, :, 2], first_image[:, :, 3]))
            #         out_image = self.net.reconstruct_image(sess, first_image)
            #         out_image = np.uint8(out_image * 255)
            #         out_stack = np.vstack((
            #             out_image[:, :, 0], out_image[:, :, 1],
            #             out_image[:, :, 2], out_image[:, :, 3]))
            #         side_by_side = np.hstack((first_stack, out_stack))
            #         file = self.folder / "sae_{0:07d}.png".format(i)
            #         cv2.imwrite(str(file), side_by_side)
            if i % 1000 == 0:
                logger.info("Classifier: training {} steps".format(i))
                log = (train_loss, sl_loss, ae_loss, acc)
                logger.info("Classifier: total_loss={}, SL_loss={}, "
                            "AE_loss={}, "
                            "Train_acc={}".format(*log))
        # # Test final network on how well it plays the actual game
        # total_reward, total_steps, n_episodes = self.test_game(sess)
        # summary.value.add(tag='Reward', simple_value=total_reward)
        # summary.value.add(tag='Steps', simple_value=total_steps)
        # summary.value.add(tag='Episodes', simple_value=n_episodes)

        # if total_reward >= self.best_model_reward:
        #     self.save_best_model(total_reward, best_saver, sess)
        classify_ctr += 1

        # if classify_ctr % 10 == 0:
        log_data = (classify_ctr, len(exp_buffer))
        logger.info("Classifier: #updates={} "
                    "exp_buffer_size={}".format(*log_data))

        return classify_ctr


def ae_classify_demo(args):
    """Use Autoencoder to learn features and classify demo."""
    GYM_ENV_NAME = args.gym_env.replace('-', '_')

    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        assert args.cuda_devices != ''
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    import tensorflow as tf

    if args.cpu_only:
        device = "/cpu:0"
        gpu_options = None
    else:
        device = "/gpu:"+os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=args.gpu_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False)

    if args.demo_memory_folder is not None:
        demo_memory_folder = args.demo_memory_folder
    else:
        demo_memory_folder = 'collected_demo/{}'.format(GYM_ENV_NAME)

    demo_memory_folder = pathlib.Path(demo_memory_folder)

    args.use_mnih_2015 = True  # ONLY supports this network
    if args.model_folder is not None:
        model_folder = '{}_{}'.format(GYM_ENV_NAME, args.model_folder)
    else:
        model_folder = 'results/pretrain_models/{}'.format(GYM_ENV_NAME)
        end_str = ''
        if args.use_mnih_2015:
            end_str += '_mnih2015'
        if args.padding == 'SAME':
            end_str += '_same'
        if args.optimizer == 'adam':
            end_str += '_adam'
        if args.exclude_noop:
            end_str += '_exclude_noop'
        if args.exclude_num_demo_ep > 0:
            end_str += '_exclude{}demoeps'.format(args.exclude_num_demo_ep)
        if args.l2_beta > 0:
            end_str += '_l2beta{:.0E}'.format(args.l2_beta)
        if args.l1_beta > 0:
            end_str += '_l1beta{:.0E}'.format(args.l1_beta)
        if args.grad_norm_clip is not None:
            end_str += '_clipnorm{:.0E}'.format(args.grad_norm_clip)
        if args.sampling_type is not None:
            end_str += '_{}'.format(args.sampling_type)
        if args.sae_classify_demo:
            end_str += '_sae'
            args.ae_classify_demo = False
        else:
            end_str += '_ae'
            args.sae_classify_demo = False
        if args.use_slv:
            end_str += '_slv'
        if args.sl_loss_weight < 1:
            end_str += '_slweight{:.0E}'.format(args.sl_loss_weight)
        if args.use_denoising:
            end_str += '_noise{:.0E}'.format(args.noise_factor)
        if args.tied_weights:
            end_str += '_tied'
        if args.loss_function == 'bce':
            end_str += '_bce'
        else:
            end_str += '_mse'
        model_folder += end_str

    if args.append_experiment_num is not None:
        model_folder += '_' + args.append_experiment_num

    model_folder = pathlib.Path(model_folder)

    if not (model_folder / 'transfer_model').exists():
        os.makedirs(str(model_folder / 'transfer_model'))
        os.makedirs(str(model_folder / 'transfer_model/all'))
        os.makedirs(str(model_folder / 'transfer_model/nofc2'))
        os.makedirs(str(model_folder / 'transfer_model/nofc1'))
        if args.use_mnih_2015:
            os.makedirs(str(model_folder / 'transfer_model/noconv3'))
        os.makedirs(str(model_folder / 'transfer_model/noconv2'))
        os.makedirs(str(model_folder / 'model_best'))

    fh = logging.FileHandler(str(model_folder / 'classify.log'), mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = LogFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logging.getLogger('atari_wrapper').addHandler(fh)
    logging.getLogger('network').addHandler(fh)
    logging.getLogger('deep_rl').addHandler(fh)
    logging.getLogger('replay_memory').addHandler(fh)

    game_state = GameState(env_id=args.gym_env)
    action_size = game_state.env.action_space.n

    AutoEncoderNetwork.use_mnih_2015 = True  # ONLY supports mnih_2015
    AutoEncoderNetwork.l1_beta = args.l1_beta
    AutoEncoderNetwork.l2_beta = args.l2_beta
    AutoEncoderNetwork.use_gpu = not args.cpu_only
    network = AutoEncoderNetwork(
        action_size, -1, device, padding=args.padding,
        in_shape=(args.input_shape, args.input_shape, 4),
        sae=args.sae_classify_demo, tied_weights=args.tied_weights,
        use_denoising=args.use_denoising, noise_factor=args.noise_factor,
        loss_function=args.loss_function, use_slv=args.use_slv)

    logger.info("optimizer: {}".format(
        'RMSPropOptimizer' if args.optimizer == 'rms' else 'AdamOptimizer'))
    logger.info("\tlearning_rate: {}".format(args.learn_rate))
    logger.info("\tepsilon: {}".format(args.opt_epsilon))
    if args.optimizer == 'rms':
        logger.info("\tdecay: {}".format(args.opt_alpha))
    else:  # Adam
        # Tensorflow defaults
        beta1 = 0.9
        beta2 = 0.999

    with tf.device(device):
        ae_opt = None

        if args.optimizer == 'rms':
            if args.ae_classify_demo:
                ae_opt = tf.train.RMSPropOptimizer(
                    learning_rate=args.learn_rate,
                    decay=args.opt_alpha,
                    epsilon=args.opt_epsilon,
                    )
            opt = tf.train.RMSPropOptimizer(
                learning_rate=args.learn_rate,
                decay=args.opt_alpha,
                epsilon=args.opt_epsilon,
                )

        else:  # Adam
            if args.ae_classify_demo:
                ae_opt = tf.train.AdamOptimizer(
                    learning_rate=args.learn_rate,
                    beta1=beta1, beta2=beta2,
                    epsilon=args.opt_epsilon,
                    )
            opt = tf.train.AdamOptimizer(
                learning_rate=args.learn_rate,
                beta1=beta1, beta2=beta2,
                epsilon=args.opt_epsilon,
                )

    ae_classify_demo = AutoencoderClassifyDemo(
        tf, network, args.gym_env, int(args.train_max_steps),
        args.batch_size, ae_opt, opt, eval_freq=args.eval_freq,
        demo_memory_folder=demo_memory_folder,
        demo_ids=args.demo_ids,
        folder=model_folder,
        exclude_num_demo_ep=args.exclude_num_demo_ep,
        device=device, clip_norm=args.grad_norm_clip,
        game_state=game_state,
        sampling_type=args.sampling_type,
        sl_loss_weight=args.sl_loss_weight,
        reward_constant=args.reward_constant,
        )

    # prepare session
    sess = tf.Session(config=config, graph=network.graph)

    with network.graph.as_default():
        init = tf.global_variables_initializer()
    sess.run(init)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(str(model_folder / 'log'),
                                           sess.graph)

    # init or load checkpoint with saver
    with network.graph.as_default():
        saver = tf.train.Saver()
        best_saver = tf.train.Saver(max_to_keep=1)

    def signal_handler(signal, frame):
        nonlocal ae_classify_demo
        logger.info('You pressed Ctrl+C!')
        ae_classify_demo.stop_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C to stop')

    if args.ae_classify_demo:
        ae_classify_demo.train_autoencoder(sess, summary_op, summary_writer)

    # else:
    ae_classify_demo.train(sess, summary_op, summary_writer,
                           best_saver=best_saver)

    logger.info('Now saving data. Please wait')
    saver.save(sess, str(model_folder
               / '{}_checkpoint'.format(GYM_ENV_NAME)))

    with network.graph.as_default():
        transfer_params = tf.get_collection("transfer_params")
        transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(
        sess, str(model_folder / 'transfer_model/all'
                  / '{}_transfer_params'.format(GYM_ENV_NAME)))

    # Remove fc2/fc3 weights
    for param in transfer_params[:]:
        name = param.op.name
        if name == "net_-1/fc2_weights" or name == "net_-1/fc2_biases":
            transfer_params.remove(param)
        elif name == "net_-1/fc3_weights" or name == "net_-1/fc3_biases":
            transfer_params.remove(param)

    with network.graph.as_default():
        transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(
        sess, str(model_folder / 'transfer_model/nofc2'
                  / '{}_transfer_params'.format(GYM_ENV_NAME)))

    # Remove fc1 weights
    for param in transfer_params[:]:
        name = param.op.name
        if name == "net_-1/fc1_weights" or name == "net_-1/fc1_biases":
            transfer_params.remove(param)

    with network.graph.as_default():
        transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(
        sess, str(model_folder / 'transfer_model/nofc1'
                  / '{}_transfer_params'.format(GYM_ENV_NAME)))

    # Remove conv3 weights
    if args.use_mnih_2015:
        for param in transfer_params[:]:
            name = param.op.name
            if name == "net_-1/conv3_weights" or name == "net_-1/conv3_biases":
                transfer_params.remove(param)

        with network.graph.as_default():
            transfer_saver = tf.train.Saver(transfer_params)
        transfer_saver.save(
            sess, str(model_folder / 'transfer_model/noconv3'
                      / '{}_transfer_params'.format(GYM_ENV_NAME)))

    # Remove conv2 weights
    for param in transfer_params[:]:
        name = param.op.name
        if name == "net_-1/conv2_weights" or name == "net_-1/conv2_biases":
            transfer_params.remove(param)

    with network.graph.as_default():
        transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(
        sess, str(model_folder / 'transfer_model/noconv2'
                  / '{}_transfer_params'.format(GYM_ENV_NAME)))

    # if args.sae_classify_demo:
    max_output_value_file = model_folder / 'transfer_model/max_output_value'
    with max_output_value_file.open('w') as f:
        f.write(str(ae_classify_demo.max_val))

    logger.info('Data saved!')
    sess.close()
