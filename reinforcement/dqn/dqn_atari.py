import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all messages
import numpy as np
import gym
import time
import json
from collections import deque  # for replay memory

from utils import create_directories
from atari import rgb_to_grayscale, collect_frames
from queues import Queue

tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', 'train', """'Train' or 'test'.""")
tf.app.flags.DEFINE_string('env_name', 'Pong-v0', """Gym environment.""")
tf.app.flags.DEFINE_string('device', '/cpu:0', """'/cpu:0' or '/gpu:0'.""")
tf.app.flags.DEFINE_float('learning_rate', 0.00025, """Initial learning rate.""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Examples per training update.""")
tf.app.flags.DEFINE_float('discount_factor', 0.99, """Discount factor in update.""")
tf.app.flags.DEFINE_float('init_epsilon', 1.0, """Initial exploration rate.""")
tf.app.flags.DEFINE_float('min_epsilon', 0.05, """Minimum exploration rate.""")
tf.app.flags.DEFINE_integer('anneal_steps', 1000000, """Steps to anneal exploration over.""")
tf.app.flags.DEFINE_integer('episodes', 10000, """Episodes per train/test run.""")
tf.app.flags.DEFINE_integer('update_freq', 4, """Number of actions between updates.""")
tf.app.flags.DEFINE_integer('agent_history', 4, """Number of frames per state.""")
tf.app.flags.DEFINE_integer('clone_steps', 10000, """Steps between cloning ops.""")
tf.app.flags.DEFINE_integer('max_steps', 100000, """Maximum steps per episode.""")
tf.app.flags.DEFINE_integer('min_memory_size', 10000, """Minimum number of replay memories.""")
tf.app.flags.DEFINE_integer('max_memory_size', 100000, """Maximum number of replay memories.""")
tf.app.flags.DEFINE_integer('ckpt_freq', 25, """Episodes per checkpoint.""")
tf.app.flags.DEFINE_integer('log_freq', 25, """Steps per log.""")
tf.app.flags.DEFINE_string('base_dir', '.', """Base directory for checkpoints and logs.""")
tf.app.flags.DEFINE_boolean('render', False, """Render episodes (once per `ckpt_freq` in training mode).""")


def available_actions(env):
    try:
        return env.action_space.n
    except AttributeError:
        raise AttributeError("env.action_space is not Discrete")

def cnn(x, n_actions, scope=''):

    with tf.variable_scope(scope):

        # conv1
        x = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=8,
            strides=4,
            padding='valid',
            activation=tf.nn.relu)
        # tf.summary.histogram('conv1', x)

        # conv2
        x = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=4,
            strides=2,
            padding='valid',
            activation=tf.nn.relu)
        # tf.summary.histogram('conv2', x)

        # conv3
        x = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=3,
            strides=1,
            padding='valid',
            activation=tf.nn.relu)
        # tf.summary.histogram('conv3', x)

        # dense
        x = tf.layers.dense(tf.reshape(x, (-1, 64 * 7 * 7)), 512, tf.nn.relu)
        # tf.summary.histogram('dense', x)

        # logits
        logits = tf.layers.dense(x, n_actions)
        # tf.summary.histogram('logits', logits)
        return logits

def sample_memory(memory, size=32):
    """Sample `size` transitions from `memory` uniformly"""
    idx = np.random.choice(range(len(memory.queue)), size)
    batch = [memory.queue[i] for i in idx]
    states = np.array([b[0] for b in batch])
    actions = np.array([b[1] for b in batch])
    rewards = np.clip(np.array([b[2] for b in batch]), -1, 1)  # reward clipping
    next_states = np.array([b[3] for b in batch])
    dones = np.array([b[4] for b in batch])
    return states, actions, rewards, next_states, dones

def log_scalar(writer, tag, value, step):
    value = [tf.Summary.Value(tag=tag, simple_value=value)]
    summary = tf.Summary(value=value)
    writer.add_summary(summary, step)

def train(env_name='CartPole-v0',
          device='/cpu:0',
          learning_rate=1e-3,
          batch_size=32,
          discount_factor=0.99,
          init_epsilon=1.0,
          min_epsilon=0.05,
          anneal_steps=1000000,
          episodes=10000,
          update_freq=4,
          agent_history=4,
          clone_steps=10000,
          max_steps=100000,
          min_memory_size=10000,
          max_memory_size=100000,
          ckpt_freq=25,
          log_freq=25,
          base_dir=None,
          render=True):

    # create log and checkpoint directories
    if base_dir is not None:
        ckpt_dir, log_dir, meta_dir = create_directories(env_name, "dqn_atari", base_dir=base_dir)
        meta = {
            'env_name': env_name,
            'device': device,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'discount_factor': discount_factor,
            'init_epsilon': init_epsilon,
            'min_epsilon': min_epsilon,
            'anneal_steps': anneal_steps,
            'episodes': episodes,
            'update_freq': update_freq,
            'agent_history': agent_history,
            'clone_steps': clone_steps,
            'max_steps': max_steps,
            'min_memory_size': min_memory_size,
            'max_memory_size': max_memory_size,
        }
        with open(meta_dir + '/meta.json', 'w') as file:
            json.dump(meta, file, indent=2)
    else:
        ckpt_dir = log_dir = None

    # create an environment
    env = gym.make(env_name)
    n_actions = available_actions(env)

    # construct graph
    with tf.device(device):

        print('constructing graph on device: {}'.format(device))

        # create placeholders
        states_pl = tf.placeholder(tf.uint8, (None, 84, 84, agent_history))
        actions_pl = tf.placeholder(tf.int32, (None, ))
        targets_pl = tf.placeholder(tf.float32, (None, ))

        # create networks
        action_values = cnn(
            tf.cast(states_pl, tf.float32) / 255.0,
            n_actions,
            scope='value')
        target_values = cnn(
            tf.cast(states_pl, tf.float32) / 255.0,
            n_actions,
            scope='target')

        # action selection
        greedy_action = tf.arg_max(action_values, dimension=1)
        target_actions = tf.arg_max(target_values, dimension=1)

        # action-value calculation
        value_mask = tf.one_hot(actions_pl, n_actions)
        target_mask = tf.one_hot(target_actions, n_actions)
        values = tf.reduce_sum(value_mask * action_values, axis=1)
        targets = tf.reduce_sum(target_mask * target_values, axis=1)

        # define training operation
        loss = tf.clip_by_value(tf.losses.mean_squared_error(values, targets_pl), -1, 1)  # error clipping
        train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                             momentum=0.95,
                                             epsilon=0.01).minimize(loss)

        # define cloning operation
        source = tf.get_default_graph().get_collection('trainable_variables', scope='value')
        target = tf.get_default_graph().get_collection('trainable_variables', scope='target')
        clone_ops = [tf.assign(t, s, name='clone') for t,s in zip(target, source)]

    # create preprocessing ops
    frame_pl = tf.placeholder(tf.uint8, [210, 160, 3])
    preprocess_op = rgb_to_grayscale(frame_pl)  # uint8, 84 x 84

    # create summary ops
    imgs = tf.transpose(tf.reshape(states_pl[0, :, :, :], [-1, 84, 84, agent_history]), perm=[3, 1, 2, 0])
    tf.summary.image('state', imgs, max_outputs=4)
    tf.summary.histogram('action_values', values)
    tf.summary.scalar('loss', loss)
    summary_op = tf.summary.merge_all()

    # create a saver
    saver = tf.train.Saver()

    def preprocess(frame, sess):
        return sess.run(preprocess_op, {frame_pl: frame})

    def create_replay_memory(env, sess, agent_history, min_memory_size, max_memory_size):
        """Initialize replay memory to `size`. Collect experience under random policy."""

        print('Creating replay memory...')
        t0 = time.time()
        memory_queue = Queue(size=max_memory_size)
        while len(memory_queue) < min_memory_size:
            frame = env.reset()
            frame_queue = Queue(init_values=[preprocess(frame, sess)], size=agent_history)
            state = collect_frames(frame_queue, nframes=agent_history)
            while True:
                action = np.random.randint(env.action_space.n)
                next_frame, reward, done, info = env.step(action)
                frame_queue.push(preprocess(next_frame, sess))
                next_state = collect_frames(frame_queue, nframes=agent_history)
                memory_queue.push([state, action, reward, next_state, done])
                state = next_state.copy()
                if done or len(memory_queue) == min_memory_size:
                    break
        elapsed_time = time.time() - t0
        print('done (elapsed time: {:.2f})'.format(elapsed_time))
        return memory_queue

    def clone_network(sess):
        """Clone `action_values` network to `target_values`."""
        sess.run(clone_ops)

    def anneal_epsilon(step, init_epsilon=1.0, min_epsilon=0.05, anneal_steps=1000000):
        """
            Linear annealing of `init_epsilon` to `min_epsilon` over `anneal_steps`.

            (Default `init_epsilon` and `min_epsilon` are the same as Mnih et al. 2015).
        """

        epsilon = init_epsilon - step * (init_epsilon - min_epsilon) / anneal_steps
        return np.maximum(min_epsilon, epsilon)

    def run_episode(env, sess, global_step, global_epsilon, render=False, delay=0.0):
        frame = env.reset()
        frame_queue = Queue(init_values=[preprocess(frame, sess)], size=agent_history)
        state = collect_frames(frame_queue, nframes=agent_history)
        episode_reward = 0
        episode_steps = 0
        if render == True:
            env.render()
            time.sleep(delay)
        while True:

            # select an action
            if np.random.rand() < global_epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = sess.run(greedy_action,
                    feed_dict={
                        states_pl: state.reshape(1, 84, 84, agent_history)
                    })
                action = action[0]

            # perform action
            next_frame, reward, done, info = env.step(action)
            if render == True:
                env.render()
                time.sleep(delay)

            # calculate next state
            frame_queue.push(preprocess(next_frame, sess))
            next_state = collect_frames(frame_queue, nframes=agent_history)

            # save transition to memory
            memory_queue.push([state, action, reward, next_state, done])
            state = next_state.copy()

            # update episode totals
            episode_reward += reward
            episode_steps += 1

            # update global step count
            global_step += 1

            # update network
            if episode_steps % update_freq == 0:

                # calculate target action-values
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = sample_memory(memory_queue, size=batch_size)
                batch_targets = sess.run(targets,
                    feed_dict={
                        states_pl: batch_next_states,
                    })
                batch_targets = batch_rewards + ~batch_dones * discount_factor * batch_targets

                # parameter update
                if episode_steps % log_freq == 0:
                    # ... w/ logging...
                    summary, _ = sess.run([summary_op, train_op],
                        feed_dict={
                            states_pl: batch_states,
                            actions_pl: batch_actions,
                            targets_pl: batch_targets,
                        })
                    writer.add_summary(summary, global_step)
                else:
                    # ... w/o logging...
                    sess.run(train_op,
                        feed_dict={
                            states_pl: batch_states,
                            actions_pl: batch_actions,
                            targets_pl: batch_targets,
                        })

            # update target network
            if global_step % clone_steps == 0:
                sess.run(clone_ops)

            # update global epsilon
            global_epsilon = anneal_epsilon(global_step,
                                            init_epsilon,
                                            min_epsilon,
                                            anneal_steps)

            # check if episode is done
            if done:
                break
            elif episode_steps == max_steps:
                print("episode reached max_steps")
                break

        return episode_reward, episode_steps, global_step, global_epsilon

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(clone_ops)  # set networks equal to begin
        memory_queue = create_replay_memory(env, sess, agent_history, min_memory_size, max_memory_size)
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        global_step = 0
        global_epsilon = init_epsilon
        t0 = time.time()
        reward_history = []
        for episode in range(episodes):

            # run an episode
            if (episode + 1) % ckpt_freq == 0:
                reward, steps, global_step, global_epsilon = run_episode(env, sess, global_step, global_epsilon, render=render)  # optionally render one episode per checkpoint
            else:
                reward, steps, global_step, global_epsilon = run_episode(env, sess, global_step, global_epsilon, render=False)
            reward_history += [reward]
            elapsed_time = time.time() - t0
            avg_reward = np.mean(reward_history[-100:])
            print('episode: {:d},  reward: {:.2f},  avg. reward: {:.2f},  steps:  {:d},  epsilon: {:.2f}, lr: {:.2e},  elapsed: {:.2f}'.format(episode + 1, reward, avg_reward, global_step, global_epsilon, learning_rate, elapsed_time))

            # logging
            log_scalar(writer, 'reward', reward, global_step)
            log_scalar(writer, 'steps', steps, global_step)
            log_scalar(writer, 'avg_reward', avg_reward, global_step)
            log_scalar(writer, 'learning_rate', learning_rate, global_step)
            log_scalar(writer, 'epsilon', global_epsilon, global_step)

            # checkpoint
            if (episode + 1) % ckpt_freq == 0:
                if ckpt_dir is not None:
                    saver.save(sess, save_path=ckpt_dir + "/ckpt", global_step=global_step)

        # final checkpoint
        if ckpt_dir is not None:
            saver.save(sess, save_path=ckpt_dir + "/ckpt", global_step=global_step)

def find_latest_checkpoint(ckpt_dir, prefix="ckpt-"):
    """Find the latest checkpoint in dir at `load_path` with prefix `prefix`
        E.g. ./checkpoints/CartPole-v0/dqn-vanilla/GLOBAL_STEP
    """
    files = os.listdir(ckpt_dir)
    matches = [f for f in files if f.find(prefix) == 0]  # files starting with prefix
    max_steps = np.max(np.unique([int(m.strip(prefix).split('.')[0]) for m in matches]))
    latest_checkpoint = ckpt_dir + prefix + str(max_steps)
    return latest_checkpoint

def test(env_name='CartPole-v0',
         hidden_units=[32],
         epsilon=0.01,
         episodes=100,
         restore_dir=None,
         render=False):

    # create an environment
    env = gym.make(env_name)
    n_dims = state_dimensions(env)
    n_actions = available_actions(env)

    # create placeholders
    states_pl = tf.placeholder(tf.float32, (None, n_dims))
    actions_pl = tf.placeholder(tf.int32, (None, ))

    # initialize networks
    action_values = mlp(states_pl, hidden_units + [n_actions], scope='value')
    # greedy_action = tf.math.argmax(action_values, axis=1)
    greedy_action = tf.arg_max(action_values, dimension=1)
    value_mask = tf.one_hot(actions_pl, n_actions)
    values = tf.reduce_sum(value_mask * action_values, axis=1)

    # create a saver
    saver = tf.train.Saver()

    def run_episode(env, sess, epsilon, render=False, delay=0.01):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        if render == True:
            env.render()
            time.sleep(delay)
        while True:
            # select an action
            if np.random.rand() < epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = sess.run(greedy_action, feed_dict={states_pl: state.reshape(1, -1)})[0]

            # take a step
            state, reward, done, info = env.step(action)
            # print("action: {:d}".format(action))
            if render == True:
                env.render()
                time.sleep(delay)

            # update episode totals
            episode_reward += reward
            episode_steps += 1

            # check if you're done
            if done:
                break

        return episode_reward, episode_steps

    with tf.Session() as sess:
        saver.restore(sess, find_latest_checkpoint(restore_dir))
        rewards = 0
        for i in range(episodes):
            episode_rewards, _ = run_episode(env, sess, epsilon, render=render)
            rewards += episode_rewards
        return rewards / episodes

if __name__ == '__main__':
    if FLAGS.mode == 'train':
        train(env_name=FLAGS.env_name,
              device=FLAGS.device,
              learning_rate=FLAGS.learning_rate,
              batch_size=FLAGS.batch_size,
              discount_factor=FLAGS.discount_factor,
              init_epsilon=FLAGS.init_epsilon,
              min_epsilon=FLAGS.min_epsilon,
              anneal_steps=FLAGS.anneal_steps,
              episodes=FLAGS.episodes,
              update_freq=FLAGS.update_freq,
              agent_history=FLAGS.agent_history,
              clone_steps=FLAGS.clone_steps,
              max_steps=FLAGS.max_steps,
              min_memory_size=FLAGS.min_memory_size,
              max_memory_size=FLAGS.max_memory_size,
              ckpt_freq=FLAGS.ckpt_freq,
              log_freq=FLAGS.log_freq,
              base_dir=FLAGS.base_dir,
              render=FLAGS.render)
    else:
        test(env_name=FLAGS.env_name,
             hidden_units=[int(i) for i in FLAGS.hidden_units.split(',')],
             epsilon=FLAGS.min_epsilon,
             episodes=FLAGS.episodes,
             restore_dir=FLAGS.base_dir,
             render=FLAGS.render)
