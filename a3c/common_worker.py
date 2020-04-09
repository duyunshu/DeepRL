import cv2
import logging
import tensorflow as tf
import numpy as np

from common.replay_memory import ReplayMemory
from common.util import generate_image_for_cam_video
from common.util import grad_cam
from common.util import make_movie
from common.util import visualize_cam
from common.game_state import get_wrapper_by_name
from termcolor import colored

logger = logging.getLogger("common_worker")

class CommonWorker(object):
    is_classify_thread = False
    is_sil_thread = False
    is_rollout_thread = False
    thread_idx = -1
    action_size = -1
    reward_type = 'CLIP' # CLIP | LOG | RAW
    env_id = None
    reward_constant = 0
    max_global_time_step=0

    def pick_action(self, logits):
        """Choose action probabilistically.

        Reference:
        https://github.com/ppyht2/tf-a2c/blob/master/src/policy.py
        """
        noise = np.random.uniform(0, 1, np.shape(logits))
        return np.argmax(logits - np.log(-np.log(noise)))

    def choose_action_with_high_confidence(self, pi_values, exclude_noop=True):
        """Choose action with confidence."""
        max_confidence_action = np.argmax(pi_values[1 if exclude_noop else 0:])
        confidence = pi_values[max_confidence_action]
        return (max_confidence_action+(1 if exclude_noop else 0)), confidence

    def egreedy_action(self, pi_values, epsilon=0.01):
        """Choose action with egreedy exploration."""
        rand = np.random.random()
        if rand > epsilon:
            action = np.argmax(pi_values)
        else:
            action = np.random.randint(len(pi_values))
        return action

    def set_start_time(self, start_time):
        """Set start time."""
        self.start_time = start_time

    def set_summary_writer(self, writer):
        """Set summary writer."""
        self.writer = writer

    def _anneal_learning_rate(self, global_time_step, initial_learning_rate):
        learning_rate = initial_learning_rate * \
        (self.max_global_time_step - global_time_step) / \
        self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate


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

    def testing(self, sess, max_steps, global_t, folder, demo_memory_cam=None,
                worker=None):
        """Evaluate A3C."""
        assert worker is not None
        assert not worker.is_classify_thread
        assert not worker.is_rollout_thread
        assert not worker.is_sil_thread

        logger.info("Evaluate policy at global_t={}...".format(global_t))

        # copy weights from shared to local
        sess.run(worker.sync)

        if demo_memory_cam is not None and global_t % 5000000 == 0:
            worker.generate_cam_video(sess, 0.03, global_t, folder,
                                    demo_memory_cam)

        episode_buffer = []
        worker.game_state.reset(hard_reset=True)
        episode_buffer.append(worker.game_state.get_screen_rgb())

        total_reward = 0
        total_steps = 0
        episode_reward = 0
        episode_steps = 0
        n_episodes = 0
        while max_steps > 0:
            state = cv2.resize(worker.game_state.s_t,
                               worker.local_net.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)
            pi_, value_, logits_ = worker.local_net.run_policy_and_value(
                sess, state)

            if False:
                action = np.random.choice(range(worker.action_size), p=pi_)
            else:
                action = worker.pick_action(logits_)

            # take action
            worker.game_state.step(action)
            terminal = worker.game_state.terminal

            if n_episodes == 0 and global_t % 5000000 == 0:
                episode_buffer.append(worker.game_state.get_screen_rgb())

            episode_reward += worker.game_state.reward
            episode_steps += 1
            max_steps -= 1

            # s_t = s_t1
            worker.game_state.update()

            if terminal:
                env = worker.game_state.env
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
                    log_data = (global_t, worker.thread_idx, self.thread_idx,
                                n_episodes, score_str, steps_str,
                                total_steps)
                    logger.debug("test: global_t={} test_worker={} cur_worker={}"
                                 " trial={} {} {}"
                                 " total_steps={}".format(*log_data))
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
                    total_reward, total_steps,
                    n_episodes)
        logger.info("test: global_t={} test_worker={} cur_worker={}"
                    " final score={} final steps={}"
                    " # trials={}".format(*log_data))

        worker.record_summary(
            score=total_reward, steps=total_steps,
            episodes=n_episodes, global_t=global_t, mode='A3C_Test')

        # reset variables used in training
        worker.episode_reward = 0
        worker.episode_steps = 0
        worker.game_state.reset(hard_reset=True)
        worker.last_rho = 0.

        if worker.use_sil:
            # ensure no states left from a non-terminating episode
            worker.episode.reset()
        return (total_reward, total_steps, n_episodes)

    def generate_cam(self, sess, test_cam_si, global_t, worker=None):
        """Compute Grad-CAM and generate video of Grad-CAM."""
        assert worker is not None

        cam_side_img = []

        for i in range(len(test_cam_si)):
            state = test_cam_si[i]

            if np.isnan(state).any():
                logger.error("Nan in state_img {}".format(i))
                continue

            state = cv2.resize(state, worker.local_net.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)

            # get max action per demo state
            readout_t = worker.local_net.run_policy(sess, state)
            action = np.argmax(readout_t)

            # convert action to one-hot vector
            action_onehot = [0.] * worker.game_state.env.action_space.n
            action_onehot[action] = 1.

            # compute grad cam for conv layer 3
            activations, gradients = worker.local_net.evaluate_grad_cam(
                sess, state, action_onehot)
            cam = grad_cam(activations, gradients)
            cam_img = visualize_cam(cam, shape=worker.local_net.in_shape[:-1])

            side_by_side = generate_image_for_cam_video(
                state,
                cam_img, global_t, i,
                worker.action_meaning[action], shape=worker.local_net.in_shape[0])

            cam_side_img.append(side_by_side)

        return cam_side_img

    def generate_cam_video(self, sess, time_per_step, global_t, folder,
                           demo_memory_cam, demo_cam_human=False, worker=None):
        """Generate the Grad-CAM video."""
        assert worker is not None

        cam_side_img = self.generate_cam(sess, demo_memory_cam, global_t, worker)

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
                      demo_memory_cam=None, demo_cam_human=False,
                      worker=None):
        """Test A3C model and generate Grad-CAM."""
        assert worker is not None

        logger.info("Testing model at global_t={}...".format(global_t))
        # copy weights from shared to local
        sess.run(worker.sync)

        if demo_memory_cam is not None:
            self.generate_cam_video(sess, 0.03, global_t, folder,
                                    demo_memory_cam, demo_cam_human, worker)
            return
        else:
            worker.game_state.reset(hard_reset=True)
            max_steps += 4
            # print(self.game_state.clone_full_state().shape)
            # self.game_state.env.render()
            # int(input())
            test_memory = ReplayMemory(
                84, 84,  # default even if input shape is different
                np.random.RandomState(),
                max_steps=max_steps,
                phi_length=4,
                num_actions=worker.game_state.env.action_space.n,
                wrap_memory=False,
                full_state_size=worker.game_state.clone_full_state().shape[0])
            for _ in range(4):
                test_memory.add(
                    worker.game_state.x_t,
                    0,
                    worker.game_state.reward,
                    worker.game_state.terminal,
                    worker.game_state.lives,
                    fullstate=worker.game_state.full_state)

        episode_buffer = []
        test_memory_cam = []

        total_reward = 0
        total_steps = 0
        episode_reward = 0
        episode_steps = 0
        terminal = False
        while True:
            state = cv2.resize(worker.game_state.s_t,
                               worker.local_net.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)
            test_memory_cam.append(state)
            episode_buffer.append(worker.game_state.get_screen_rgb())
            pi_, value_, logits_ = worker.local_net.run_policy_and_value(
                sess, state)
            action = np.argmax(pi_)

            worker.game_state.step(action)
            terminal = worker.game_state.terminal
            memory_full = episode_steps == max_steps-5
            terminal_ = terminal or memory_full

            # store the transition to replay memory
            test_memory.add(
                worker.game_state.x_t1, action,
                worker.game_state.reward, terminal_,
                worker.game_state.lives,
                fullstate=worker.game_state.full_state1)

            # update the old values
            episode_reward += worker.game_state.reward
            episode_steps += 1

            # s_t = s_t1
            worker.game_state.update()

            if terminal_:
                env = worker.game_state.env
                name = 'EpisodicLifeEnv'
                if get_wrapper_by_name(env, name).was_real_done or memory_full:
                    time_per_step = 0.03
                    images = np.array(episode_buffer)
                    file = 'frames/image{ep:010d}'.format(ep=global_t)
                    duration = len(images)*time_per_step
                    make_movie(images, str(folder / file), duration=duration,
                               true_image=True, salience=False)
                    break

                worker.game_state.reset(hard_reset=False)

        total_reward = episode_reward
        total_steps = episode_steps
        log_data = (global_t, worker.thread_idx, total_reward, total_steps)
        logger.info("test: global_t={} worker={} final score={}"
                    " final steps={}".format(*log_data))

        self.generate_cam_video(sess, 0.03, global_t, folder,
                                np.array(test_memory_cam), worker)
        test_memory.save(name='test_cam', folder=folder, resize=True)

        return

    def test_retrain_classifier(self, global_t, max_steps, sess, worker=None):
        """Evaluate game with current classifier model."""
        assert worker is not None
        assert worker.is_classify_thread

        logger.info("Testing (retrained) classifier at global_t={}...".format(global_t))

        worker.game_state.reset(hard_reset=True)

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

            action, _ = self.choose_action_with_high_confidence(
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
        logger.info("(retrained) classifier test: global_t={} test_worker={} cur_worker={} "
                    "final score={} final steps={} # trials={}"
                    .format(*log_data))
        self.record_summary(
            score=total_reward, steps=total_steps,
            episodes=n_episodes, global_t=global_t, mode='Classifier_Test')

        return (total_reward, total_steps, n_episodes)


    def test_fixed_classifier(self, global_t, max_steps, sess, worker=None, model=None):
        """Evaluate game with current classifier model."""
        assert model is not None
        assert sess is not None
        assert worker is not None

        logger.info("Testing (fixed) classifier at global_t={}...".format(global_t))

        worker.game_state.reset(hard_reset=True)

        total_reward = 0
        total_steps = 0
        episode_reward = 0
        episode_steps = 0
        n_episodes = 0
        while max_steps > 0:
            # worker.game_state.env.render()
            state = cv2.resize(worker.game_state.s_t,
                               model.in_shape[:-1],
                               interpolation=cv2.INTER_AREA)
            model_pi = model.run_policy(sess, state)

            action, _ = self.choose_action_with_high_confidence(
                model_pi, exclude_noop=False)
            # action = self.egreedy_action(model_pi, epsilon=0.01)

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
                    logger.debug("(fixed) classifier test: trial={} {} {} "
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

        return (total_reward, total_steps, n_episodes)
