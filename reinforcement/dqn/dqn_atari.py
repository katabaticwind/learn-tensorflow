import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all messages
import numpy as np
import gym
import time

from atari import crop_frame, RGB_to_luminance, frames_to_state
from queues import Queue
from utils import create_directories

tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', 'train', """'Train' or 'test'.""")
tf.app.flags.DEFINE_string('env_name', 'Pong-v0', """Atari environment.""")
tf.app.flags.DEFINE_string('device', '/gpu:0', """'/cpu:0' or '/gpu:0'.""")

tf.app.flags.DEFINE_string('filters', '32,64,64', """Channels of feature maps.""")
tf.app.flags.DEFINE_string('size', '8,4,3', """Sizes of filters.""")
tf.app.flags.DEFINE_string('strides', '4,2,1', """Strides of filters.""")
tf.app.flags.DEFINE_string('hidden_units', '512', """Size of hidden layers.""")

tf.app.flags.DEFINE_float('lr', 2.5e-4, """Initial learning rate.""")
tf.app.flags.DEFINE_float('lr_decay', 1.00, """Learning rate decay (per episode).""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Examples per training update.""")
tf.app.flags.DEFINE_float('discount_factor', 0.99, """Reward discount factor (i.e., "gamma").""")
tf.app.flags.DEFINE_boolean('clip_rewards', True, """Clip rewards to [-1, 1].""")
tf.app.flags.DEFINE_boolean('clip_errors', True, """Clip errors to [-1, 1].""")
tf.app.flags.DEFINE_integer('clone_steps', 10000, """Steps between cloning ops.""")

tf.app.flags.DEFINE_float('init_epsilon', 1.0, """Initial exploration rate.""")
tf.app.flags.DEFINE_float('min_epsilon', 0.1, """Minimum exploration rate.""")
tf.app.flags.DEFINE_integer('anneal_steps', 1000000, """Steps per train/test routine.""")

tf.app.flags.DEFINE_integer('min_memory_size', 50000, """Minimum number of replay memories.""")
tf.app.flags.DEFINE_integer('max_memory_size', 1000000, """Maximum number of replay memories.""")

tf.app.flags.DEFINE_integer('update_freq', 4, """Frames between updates.""")
tf.app.flags.DEFINE_integer('action_repeat', 4, """Frames each action is repeated for.""")  # frames per "state"

tf.app.flags.DEFINE_integer('episodes', 10000, """Episodes per train/test routine.""")
tf.app.flags.DEFINE_integer('ckpt_freq', 10, """Episodes between status reports.""")
tf.app.flags.DEFINE_string('base_dir', '.', """Base directory for checkpoints and logs.""")
tf.app.flags.DEFINE_boolean('render', False, """Render once per batch in training mode.""")

def available_actions(env):
    # if type(env.action_space) == gym.spaces.discrete.Discrete:
    try:
        return env.action_space.n
    except AttributeError:
        raise AttributeError("env.action_space is not Discrete")

def state_dimensions(env):
    """Find the number of dimensions in the state."""
    return env.observation_space.shape[0]

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

        # conv2
        x = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=4,
            strides=2,
            padding='valid',
            activation=tf.nn.relu)

        # conv3
        x = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=3,
            strides=1,
            padding='valid',
            activation=tf.nn.relu)

        # dense
        x = tf.layers.dense(tf.reshape(x, (-1, 64 * 7 * 7)), 512, tf.nn.relu)

        # logits
        return tf.layers.dense(x, n_actions)

def create_replay_memory(env, action_repeat, min_memory_size, max_memory_size):
    """Initialize replay memory to `size`. Collect experience under random policy."""

    print('Creating replay memory...')
    t0 = time.time()
    memory_queue = Queue(max_memory_size)
    while len(memory_queue) < min_memory_size:
        frame = env.reset()
        frame_queue = Queue(action_repeat, [frame])
        state = frames_to_state(frame_queue, min_len=action_repeat)
        while True:
            action = np.random.randint(env.action_space.n)
            next_frame, reward, done, info = env.step(action)
            frame_queue.push(next_frame)
            next_state = frames_to_state(frame_queue, min_len=action_repeat)
            memory_queue.push([state, action, reward, next_state, done])
            state = next_state
            if done or len(memory_queue) == min_memory_size:
                break
    elapsed_time = time.time() - t0
    print('done (elapsed time: {:.2f})'.format(elapsed_time))
    return memory_queue

def sample_memory(memory, size):
    """Sample `size` transitions from `memory` uniformly"""
    idx = np.random.choice(range(len(memory.queue)), size)
    batch = [memory.queue[i] for i in idx]
    states = np.array([b[0] for b in batch])
    actions = np.array([b[1] for b in batch])
    rewards = np.clip(np.array([b[2] for b in batch]), -1, 1)
    next_states = np.array([b[3] for b in batch])
    dones = np.array([b[4] for b in batch])
    return {'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones}

def train(env_name='Pong-v0',
          device='/gpu:0',
          lr=2.5e-4,
          lr_decay=1.00,
          batch_size=32,
          discount_factor=0.99,
          clip_rewards=True,
          clip_errors=True,
          clone_steps=10000,
          init_epsilon=1.00,
          min_epsilon=0.1,
          anneal_steps=1000000,
          min_memory_size=50000,
          max_memory_size=1000000,
          update_freq=4,
          action_repeat=4,
          episodes=10000,
          ckpt_freq=100,
          base_dir='.',
          render=False):

    # create log and checkpoint directories
    if base_dir is not None:
        ckpt_dir, log_dir = create_directories(env_name, "dqn_atari", base_dir=base_dir)
    else:
        ckpt_dir = log_dir = None

    # create an environment
    env = gym.make(env_name)
    n_actions = available_actions(env)

    # construct graph
    with tf.device(device):

        print('constructing graph on device: {}'.format(device))

        # create placeholders
        states_pl = tf.placeholder(tf.float32, (None, 84, 84, action_repeat))
        actions_pl = tf.placeholder(tf.int32, (None, ))
        targets_pl = tf.placeholder(tf.float32, (None, ))

        # initialize networks
        estimates = cnn(states_pl, n_actions, scope='estimate')
        targets = cnn(states_pl, n_actions, scope='target')
        greedy_action = tf.argmax(estimates, axis=1)
        target_actions = tf.argmax(targets, axis=1)
        action_mask = tf.one_hot(actions_pl, n_actions)
        target_mask = tf.one_hot(target_actions, n_actions)
        action_values = tf.reduce_sum(action_mask * estimates, axis=1)
        target_values = tf.reduce_sum(target_mask * targets, axis=1)  # minus reward

        # define cloning operation
        target = tf.get_default_graph().get_collection('trainable_variables', scope='target')
        source = tf.get_default_graph().get_collection('trainable_variables', scope='estimate')
        clone_ops = [tf.assign(t, s, name='clone') for t,s in zip(target, source)]

        # define training operation
        loss = tf.losses.mean_squared_error(action_values, target_values)
        clipped_loss = tf.clip_by_value(loss, -1, 1)
        train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(clipped_loss)

    # create a saver
    saver = tf.train.Saver()

    # initialize replay memory
    memory_queue = create_replay_memory(env, action_repeat, min_memory_size, max_memory_size)

    # create a saver
    saver = tf.train.Saver()

    def clone_network(sess):
        """Clone `estimate_values` network to `target_values`."""
        sess.run(clone_ops)

    def anneal_epsilon(step, init_epsilon=1.0, min_epsilon=0.1, anneal_steps=1000000):
        """
            Linear annealing of `init_epsilon` to `min_epsilon` over `anneal_steps`.

            (Default `init_epsilon` and `min_epsilon` are the same as Mnih et al. 2015).
        """

        epsilon = init_epsilon - step * (init_epsilon - min_epsilon) / anneal_steps
        return np.maximum(min_epsilon, epsilon)

    def log_scalar(writer, tag, value, step):
        value = [tf.Summary.Value(tag=tag, simple_value=value)]
        summary = tf.Summary(value=value)
        writer.add_summary(summary, step)

    def run_episode(env, sess, global_step, global_epsilon, render=False, delay=0.01):
        frame = env.reset()
        frame_queue = Queue(action_repeat, [frame])
        state = frames_to_state(frame_queue, min_len=action_repeat)
        episode_reward = 0
        episode_steps = 0
        if render == True:
            env.render()
            time.sleep(delay)
        while True:
            if np.random.rand() < global_epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = sess.run(greedy_action, feed_dict={states_pl: state.reshape(1, 84, 84, action_repeat)})[0]

            # take a step
            next_frame, reward, done, info = env.step(action)
            if render == True:
                env.render()
                time.sleep(delay)

            # save transition to memory
            frame_queue.push(next_frame)
            next_state = frames_to_state(frame_queue, min_len=action_repeat)
            memory_queue.push([state, action, reward, next_state, done])
            state = next_state

            # update episode totals
            episode_reward += reward
            episode_steps += 1

            # update network parameters
            if episode_steps % update_freq == 0:

                # calculate target
                batch = sample_memory(memory_queue, batch_size)
                batch['targets'] = sess.run(target_values,
                    feed_dict={
                        states_pl: batch['next_states'],
                    })
                batch['targets'] = batch['rewards'] + ~batch['dones'] * discount_factor * batch['targets']

                # gradient descent
                sess.run(train_op,
                    feed_dict={
                        states_pl: batch['states'],
                        actions_pl: batch['actions'],
                        targets_pl: batch['targets'],
                    })

            # update step count
            global_step += 1

            # update target network
            if global_step % clone_steps == 0:
                sess.run(clone_ops)
                # print("cloned action-value network")

            # update epsilon
            global_epsilon = anneal_epsilon(global_step,
                                            init_epsilon,
                                            min_epsilon,
                                            anneal_steps)

            # check if done
            if done:
                break

        return episode_reward, episode_steps, global_step, global_epsilon

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(clone_ops)  # set networks equal to begin
        writer = tf.summary.FileWriter(log_dir, sess.graph)  # create writer for logging
        global_step = 0
        global_epsilon = init_epsilon
        t0 = time.time()
        reward_history = []
        for episode in range(episodes):
            if (episode + 1) % ckpt_freq == 0:
                reward, steps, global_step, global_epsilon = run_episode(env, sess, global_step, global_epsilon, render=render)  # optionally render one episode per checkpoint
            else:
                reward, steps, global_step, global_epsilon = run_episode(env, sess, global_step, global_epsilon, render=False)  # TODO: switch render arg back to False...
            lr = lr * lr_decay  # decay learning rate per episode
            reward_history += [reward]
            elapsed_time = time.time() - t0
            avg_reward = np.mean(reward_history[-100:])  # moving avg (last 100 episodes)
            print('episode: {:d},  reward: {:.2f},  avg. reward: {:.2f},  steps:  {:d},  epsilon: {:.2f}, lr: {:.2e},  elapsed: {:.2f}'.format(episode + 1, reward, avg_reward, global_step, global_epsilon, lr, elapsed_time))

            # logging
            log_scalar(writer, 'reward', reward, global_step)
            log_scalar(writer, 'learning_rate', lr, global_step)
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

def test(env_name='Pong-v0',
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
    estimate_values = mlp(states_pl, hidden_units + [n_actions], scope='value')
    greedy_action = tf.math.argmax(estimate_values, axis=1)
    value_mask = tf.one_hot(actions_pl, n_actions)
    values = tf.reduce_sum(value_mask * estimate_values, axis=1)

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
        prefix = 'dqn-atari-' + env_name
        saver.restore(sess, find_latest_checkpoint(load_path, prefix))
        rewards = 0
        for i in range(episodes):
            episode_rewards, _ = run_episode(env, sess, epsilon, render=render)
            rewards += episode_rewards
        return rewards / episodes

if __name__ == '__main__':
    if FLAGS.mode == 'train':
        train(env_name=FLAGS.env_name,
              device=FLAGS.device,
              lr=FLAGS.lr,
              lr_decay=FLAGS.lr_decay,
              batch_size=FLAGS.batch_size,
              discount_factor=FLAGS.discount_factor,
              clip_rewards=FLAGS.clip_rewards,
              clip_errors=FLAGS.clip_errors,
              clone_steps=FLAGS.clone_steps,
              init_epsilon=FLAGS.init_epsilon,
              min_epsilon=FLAGS.min_epsilon,
              anneal_steps=FLAGS.anneal_steps,
              min_memory_size=FLAGS.min_memory_size,
              max_memory_size=FLAGS.max_memory_size,
              update_freq=FLAGS.update_freq,
              action_repeat=FLAGS.action_repeat,
              episodes=FLAGS.episodes,
              ckpt_freq=FLAGS.ckpt_freq,
              base_dir=FLAGS.base_dir,
              render=FLAGS.render)
    else:
        score = test(env_name=FLAGS.env_name,
                     hidden_units=hidden_units,
                     epsilon=FLAGS.min_epsilon,
                     episodes=FLAGS.episodes,
                     load_path=FLAGS.load_path,
                     render=FLAGS.render)
        print("> mean = {:.2f}".format(score))
