#!/usr/bin/env python3
import argparse
import coloredlogs
import logging

from a3c import run_a3c
from a3c_test import run_a3c_test
from time import sleep

logger = logging.getLogger()


def main():
    fmt = "%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s"
    coloredlogs.install(level='DEBUG', fmt=fmt)
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()

    parser.add_argument('--parallel-size', type=int, default=16,
                        help='parallel thread size')
    parser.add_argument('--gym-env', type=str, default='PongNoFrameskip-v4',
                        help='OpenAi Gym environment ID')

    parser.add_argument('--save-to', type=str, default='results/',
                        help='where to save results')
    parser.add_argument('--checkpoint-buffer', action='store_true', help='save replay buffer')
    parser.set_defaults(checkpoint_buffer=False)
    parser.add_argument('--checkpoint-freq', type=int, default=1,
                        help='checkpoint frequency, default to eval-freq*checkpoint-freq')

    parser.add_argument('--local-t-max', type=int, default=20,
                        help='repeat step size')
    parser.add_argument('--rmsp-alpha', type=float, default=0.99,
                        help='decay parameter for RMSProp')
    parser.add_argument('--rmsp-epsilon', type=float, default=1e-5,
                        help='epsilon parameter for RMSProp')

    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--append-experiment-num', type=str, default=None)

    parser.add_argument('--initial-learn-rate', type=float, default=7e-4,
                        help='initial learning rate for RMSProp')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards')
    parser.add_argument('--entropy-beta', type=float, default=0.01,
                        help='entropy regularization constant')
    parser.add_argument('--max-time-step', type=float, default=10 * 10**7,
                        help='maximum time step, use to anneal learning rate')
    parser.add_argument('--max-time-step-fraction', type=float, default=1.,
                        help='ovverides maximum time step by a fraction')
    parser.add_argument('--grad-norm-clip', type=float, default=0.5,
                        help='gradient norm clipping')
    parser.add_argument('--max-ep-step', type=float, default=10000,
                        help='maximum time step for an episode (prevent breakout from stuck)')

    parser.add_argument('--eval-freq', type=int, default=1000000)
    parser.add_argument('--eval-max-steps', type=int, default=125000)

    parser.add_argument('--use-gpu', action='store_true', help='use GPU')
    parser.set_defaults(use_gpu=False)
    parser.add_argument('--gpu-fraction', type=float, default=0.4)
    parser.add_argument('--cuda-devices', type=str, default='')

    parser.add_argument('--use-mnih-2015', action='store_true',
                        help='use Mnih et al [2015] network architecture')
    parser.set_defaults(use_mnih_2015=False)

    parser.add_argument('--input-shape', type=int, default=84,
                        help='84x84 as default')
    parser.add_argument('--padding', type=str, default='VALID',
                        help='VALID or SAME')

    parser.add_argument('--log-interval', type=int, default=500)
    parser.add_argument('--performance-log-interval', type=int, default=1000)

    parser.add_argument('--use-transfer', action='store_true')
    parser.set_defaults(use_transfer=False)
    parser.add_argument('--transfer-folder', type=str, default=None)
    parser.add_argument('--not-transfer-fc2', action='store_true')
    parser.set_defaults(not_transfer_fc2=False)
    parser.add_argument('--not-transfer-fc1', action='store_true')
    parser.set_defaults(not_transfer_fc1=False)
    parser.add_argument('--not-transfer-conv3', action='store_true')
    parser.set_defaults(not_transfer_conv3=False)
    parser.add_argument('--not-transfer-conv2', action='store_true')
    parser.set_defaults(not_transfer_conv2=False)
    parser.add_argument('--finetune-upper-layers-only', action='store_true')
    parser.set_defaults(finetune_upper_layers_only=False)

    parser.add_argument('--load-memory', action='store_true')
    parser.set_defaults(load_memory=False)
    parser.add_argument('--demo-memory-folder', type=str, default=None)
    parser.add_argument('--demo-ids', type=str, default=None,
                        help='demo ids separated by comma')

    parser.add_argument('--use-grad-cam', action='store_true')
    parser.set_defaults(use_grad_cam=False)
    parser.add_argument('--load-demo-cam', action='store_true')
    parser.set_defaults(load_demo_cam=False)
    parser.add_argument('--demo-cam-id', type=str, default=None,
                        help='demo id for cam')
    parser.add_argument('--demo-cam-folder', type=str, default=None,
                        help='demo folder')

    parser.add_argument('--l2-beta', type=float, default=0.,
                        help='L2 regularization beta')
    parser.add_argument('--model-folder', type=str, default=None)

    # Alternatives to reward clipping
    parser.add_argument('--unclipped-reward', action='store_true',
                        help='use raw reward')
    parser.set_defaults(unclipped_reward=False)
    # DQfD Hester, et. al 2017
    parser.add_argument('--log-scale-reward', action='store_true',
                        help='use log scale reward from DQfD (Hester et. al)')
    parser.set_defaults(log_scale_reward=False)
    # Ape-X Pohlen, et. al 2018
    parser.add_argument('--transformed-bellman', action='store_true',
                        help='use transformed bellman equation')
    parser.set_defaults(transformed_bellman=False)
    parser.add_argument('--reward-constant', type=float, default=2.0,
                        help='value added to all non-zero rewards when using'
                             ' transformed bellman operator')

    parser.add_argument('--load-pretrained-model', action='store_true')
    parser.set_defaults(load_pretrained_model=False)
    parser.add_argument('--pretrained-model-folder', type=str, default=None)
    parser.add_argument('--classify-demo', action='store_true',
                        help='Use Supervised Classifier')
    parser.set_defaults(classify_demo=False)
    parser.add_argument('--sae-classify-demo', action='store_true',
                        help='Use Supervised Autoencoder')
    parser.set_defaults(sae_classify_demo=False)
    parser.add_argument('--ae-classify-demo', action='store_true',
                        help='Use Autoencoder')
    parser.set_defaults(ae_classify_demo=False)

    parser.add_argument('--test-model', action='store_true')
    parser.set_defaults(test_model=False)

    # sil parameters
    parser.add_argument('--use-sil', action='store_true',
                        help='self imitation learning loss (SIL)')
    parser.set_defaults(use_sil=False)
    parser.add_argument('--use-sil-neg', action='store_true',
                        help='test if also use negative samples (G-V)<0')
    parser.set_defaults(use_sil_neg=False)
    parser.add_argument('--use-sil-skip', action='store_true',
                        help='test if less #sil result in worse performace')
    parser.set_defaults(use_sil_skip=False)
    parser.add_argument('--rollout-sample-proportion', type=float, default=None,
                        help='The proportion to sample from rollout during sil, range 0-1')

    # rollout parameters
    parser.add_argument('--use-rollout', action='store_true',
                        help='use human model to correct actions')
    parser.set_defaults(use_rollout=False)
    parser.add_argument('--add-all-rollout', action='store_true',
                        help='add all rollout data, otherwise, add only when new return is better')
    parser.set_defaults(add_all_rollout=False)
    parser.add_argument('--advice-budget', type=float, default=10*10**7,
                        help='max global_steps allowed for rollout')
    parser.add_argument('--stop-rollout', action='store_true', help='rollout switch on/off')
    parser.set_defaults(stop_rollout=False)
    parser.add_argument('--nstep-bc', type=int, default=100000,
                        help='rollout using BC for n steps then thereafter follow a3c till terminal')
    parser.add_argument('--update-in-rollout', action='store_true',
                        help='make immediate update using rollout data')
    parser.set_defaults(update_in_rollout=False)
    parser.add_argument('--delay-rollout', type=int, default=0,
                        help='start rollout n-eval-freq steps late (when using self-rollout, wait till it has learned sth)')
    parser.add_argument('--one-buffer', action='store_true',
                        help='use one buffer for all workers, no separete rollout buffer')
    parser.add_argument('--roll-any', action='store_true',
                        help='rollout from any random state (not just bad ones)')
    parser.set_defaults(roll_any=False)
    parser.set_defaults(one_rollout=False)

    # train classifier parameters
    parser.add_argument('--train-classifier', action='store_true',
                        help='periodically retrain classifier')
    parser.set_defaults(train_classifier=False)
    parser.add_argument('--batch-size', type=int, default=512,
                        help='SIL batch size')
    parser.add_argument('--priority-memory', action='store_true',
                        help='Use Prioritized Replay Memory')
    parser.set_defaults(priority_memory=False)
    parser.add_argument('--memory-length', type=int, default=100000,
                        help='SIL memory size')

    # classifier parameters
    parser.add_argument('--class-train-steps', type=float, default=20000,
                        help='maximum training steps for classifier')
    parser.add_argument('--class-l1-beta', type=float, default=0.,
                        help='L1 regularization beta')
    parser.add_argument('--class-l2-beta', type=float, default=0.,
                        help='classifier L2 regularization beta')
    parser.add_argument('--class-loss-function', type=str, default='mse',
                        help='classifier mse (mean squared error) or bce (binary cross entropy)')
    parser.add_argument('--class-learn-rate', type=float, default=7e-4,
                        help='initial learning rate for RMSProp')
    parser.add_argument('--class-opt-epsilon', type=float, default=1e-5,
                        help='epsilon parameter for classifier')
    parser.add_argument('--class-optimizer', type=str, default='adam',
                        help='classifier optimizer adam)')
    parser.add_argument('--class-opt-alpha', type=float, default=0.99,
                        help='decay parameter for RMS optimizer')
    parser.add_argument('--class-sl-loss-weight', type=float, default=1.,
                        help='weighted classification loss')
    parser.add_argument('--class-use-slv', action='store_true',
                        help='supervised loss with value loss')
    parser.set_defaults(class_use_slv=False)
    parser.add_argument('--class-sample-type', type=str, default=None,
                        help='None, oversample, proportional')
    parser.add_argument('--class-tied-weights', action='store_true',
                        help='Autoencoder with tied weights')
    parser.set_defaults(tied_weights=False)
    parser.add_argument('--class-use-denoising', action='store_true')
    parser.set_defaults(use_denoising=False)
    parser.add_argument('--class-noise-factor', type=float, default=0.3)


    args = parser.parse_args()

    if args.rollout_sample_proportion is not None:
        assert args.rollout_sample_proportion > 0

    if args.test_model:
        logger.info('Testing A3C model...')
        run_a3c_test(args)
        sleep(2)
    else:
        logger.info('Running A3C...')
        sleep(2)
        run_a3c(args)


if __name__ == "__main__":
    main()
