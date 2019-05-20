import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all messages
import numpy as np
import gym
import time
from collections import deque

from utils import find_latest_checkpoint, log_scalar, create_directories

tf.logging.set_verbosity(tf.logging.ERROR)  # suppress annoying messages

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', 'train', """'Train' or 'test'.""")
tf.app.flags.DEFINE_string('device', '/cpu:0', """'/cpu:0' or '/device:GPU:0'.""")
tf.app.flags.DEFINE_string('env_name', 'CartPole-v0', """Gym environment.""")
tf.app.flags.DEFINE_string('hidden_units', '32', """Size of hidden layers.""")
tf.app.flags.DEFINE_float('learning_rate', '1e-2', """Initial learning rate.""")
tf.app.flags.DEFINE_integer('batches', 100, """Batches per training update.""")
tf.app.flags.DEFINE_integer('batch_size', 5000, """Batches per training update.""")
tf.app.flags.DEFINE_integer('episodes', 100, """Episodes per test.""")
tf.app.flags.DEFINE_integer('ckpt_freq', 10, """Batches between checkpoints.""")
tf.app.flags.DEFINE_string('base_dir', '.', """Base directory for checkpoints and logs.""")
tf.app.flags.DEFINE_boolean('render', False, """Render episodes (once per batch in training mode).""")


def available_actions(env):
    # if type(env.action_space) == gym.spaces.discrete.Discrete:
    try:
        return env.action_space.n
    except AttributeError:
        raise AttributeError("env.action_space is not Discrete")

def state_dimensions(env):
    """Find the number of dimensions in the state."""
    return env.observation_space.shape[0]

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    """Build a feedforward neural network."""
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)  # TODO: creates multiple warnings (DEPRECATED)
    return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def reward_to_go(rewards):
    """Calculate the cumulative reward at each step."""
    c = []
    for (i, r) in enumerate(reversed(rewards)):
        if i == 0:
            c += [r]
        else:
            c += [r + c[i - 1]]
    return list(reversed(c))

def train(env_name='CartPole-v0',
          device='/cpu:0',
          hidden_units=[32],
          learning_rate=1e-2,
          batches=100,
          batch_size=5000,
          ckpt_freq=10,
          base_dir=None,
          render=False):

    # create log and checkpoint directories
    if base_dir is not None:
        ckpt_dir, log_dir = create_directories(env_name, "pg_baseline", base_dir=base_dir)
    else:
        ckpt_dir = log_dir = None

    # create an environment
    env = gym.make(env_name)
    n_dims = state_dimensions(env)
    n_actions = available_actions(env)

    with tf.device(device):

        print('constructing graph on device: {}'.format(device))

        # create placeholders
        states_pl = tf.placeholder(tf.float32, (None, n_dims))
        actions_pl = tf.placeholder(tf.int32, (None, ))
        weights_pl = tf.placeholder(tf.float32, (None, ))

        # create a policy network
        logits = mlp(states_pl, hidden_units + [n_actions])
        # actions = tf.squeeze(tf.random.categorical(logits=logits, num_samples=1), axis=1)  # chooses an action
        actions = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)  # chooses an action

        # create a value network
        values = mlp(states_pl, hidden_units + [1])

        # define policy network training operation
        actions_mask = tf.one_hot(actions_pl, n_actions)
        log_probs = tf.reduce_sum(actions_mask * tf.nn.log_softmax(logits), axis=1)  # use tf.mask instead?
        policy_loss = -tf.reduce_mean((weights_pl - values) * log_probs)
        policy_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(policy_loss)  # TODO: creates tf.math_ops warning (?)

        # define value network training operation
        value_loss = tf.losses.mean_squared_error(weights_pl, tf.squeeze(values, axis=1))
        value_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(value_loss)

        # create a saver
        saver = tf.train.Saver()

        # create writer
        writer = tf.summary.FileWriter(log_dir)

    def run_episode(env, sess, render=False):
        state = env.reset()
        episode_states = [state]
        episode_actions = []
        episode_weights = []
        total_reward = 0
        total_steps = 0
        if render == True:
            env.render()
            time.sleep(0.01)
        while True:
            action = sess.run(actions, feed_dict={states_pl: state.reshape(1, -1)})  # state is (4,), state_pl requires (None, 4)
            state, reward, done, info = env.step(action[0])  # step requires scalar action
            if render == True:
                env.render()
                time.sleep(0.01)
            episode_states += [state.copy()]  # no reshape b/c we will convert to np.array later...
            episode_actions += [action[0]]
            episode_weights += [reward]
            total_reward += reward
            total_steps += 1
            if done:
                break
        return episode_states, episode_actions, reward_to_go(episode_weights), total_reward, total_steps

    def run_batch(env, sess, render=False):
        t0 = time.time()
        batch_states = []
        batch_actions = []
        batch_weights = []
        batch_rewards = 0
        batch_steps = 0
        episodes = 0
        while len(batch_weights) < batch_size:
            if episodes == 0:
                episode_states, episode_actions, episode_weights, total_reward, total_steps = run_episode(env, sess, render=render)  # render first episode *if* render = True
            else:
                episode_states, episode_actions, episode_weights, total_reward, total_steps = run_episode(env, sess, render=False)
            episodes += 1
            batch_states.extend(episode_states[:-1])  # only keep states preceeding each action
            batch_actions.extend(episode_actions)
            batch_weights.extend(episode_weights)
            batch_rewards += total_reward
            batch_steps += total_steps
        t = time.time()
        return batch_states, batch_actions, batch_weights, batch_rewards, batch_steps, episodes, t - t0

    def update_policy_network(states, actions, weights, sess):
        feed_dict = {
            states_pl: states,
            weights_pl: weights,
            actions_pl: actions
        }
        batch_loss, _ = sess.run([policy_loss, policy_train_op], feed_dict=feed_dict)
        return batch_loss

    def update_value_network(states, weights, sess):
        feed_dict = {
            states_pl: states,
            weights_pl: weights
        }
        batch_loss, _ = sess.run([value_loss, value_train_op], feed_dict=feed_dict)
        return batch_loss

    # train the network
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        global_step = 0
        for batch in range(batches):
            batch_states, batch_actions, batch_weights, batch_rewards, batch_steps, batch_episodes, batch_time = run_batch(env, sess, render)
            _ = update_value_network(batch_states, batch_weights, sess)
            _ = update_policy_network(batch_states, batch_actions, batch_weights, sess)
            global_step += batch_steps
            print("batch = {:d},  batch reward = {:.2f} (mean),  batch episodes = {:d}, batch time = {:.2f},  batch steps = {:.2f},  global steps = {:.2f}".format(batch, batch_rewards / batch_episodes, batch_episodes, batch_time, batch_steps, global_step))

            # logging
            log_scalar(writer, 'batch_reward', batch_rewards / batch_episodes, global_step)
            log_scalar(writer, 'batch_steps', batch_steps, global_step)
            log_scalar(writer, 'learning_rate', learning_rate, global_step)

            # checkpoint
            if batch % ckpt_freq == 0:
                if ckpt_dir is not None:
                    saver.save(sess, save_path=ckpt_dir + "/ckpt", global_step=global_step)

        # final checkpoint
        if ckpt_dir is not None:
            saver.save(sess, save_path=ckpt_dir + "/ckpt", global_step=global_step)
            return saver.last_checkpoints

def find_latest_checkpoint(load_path, prefix):
    """Find the latest checkpoint in dir at `load_path` with prefix `prefix`

        E.g. ./checkpoints/dqn-vanilla-CartPole-v0-GLOBAL_STEP would use find_latest_checkpoint('./checkpoints/', 'dqn-vanilla-CartPole-v0')
    """
    files = os.listdir(load_path)
    matches = [f for f in files if f.find(prefix) == 0]  # files starting with prefix
    max_steps = np.max(np.unique([int(m.strip(prefix).split('.')[0]) for m in matches]))
    latest_checkpoint = load_path + prefix + '-' + str(max_steps)
    return latest_checkpoint

def test(env_name='CartPole-v0',
         hidden_units=[32],
         episodes=100,
         restore_dir=None,
         render=False):
    """Load and test a trained model from checkpoint files."""

    # create an environment
    env = gym.make(env_name)
    n_dims = state_dimensions(env)
    n_actions = available_actions(env)

    # create placeholders
    states_pl = tf.placeholder(tf.float32, (None, n_dims))
    actions_pl = tf.placeholder(tf.int32, (None, ))
    weights_pl = tf.placeholder(tf.float32, (None, ))

    # create a policy network
    logits = mlp(states_pl, hidden_units + [n_actions])
    actions = tf.squeeze(tf.random.categorical(logits=logits, num_samples=1), axis=1)  # chooses an action

    # create saver
    saver = tf.train.Saver()

    # define core episode loop
    def run_episode(env, sess, render=False, delay=0.01):
        state = env.reset()
        episode_states = [state]
        episode_actions = []
        total_reward = 0
        if render == True:
            env.render()
            time.sleep(0.01)
        while True:
            action = sess.run(actions, feed_dict={states_pl: state.reshape(1, -1)})  # state is (4,), state_pl requires (None, 4)
            state, reward, done, info = env.step(action[0])  # step requires scalar action
            if render == True:
                env.render()
                time.sleep(0.01)
            episode_states += [state.copy()]  # no reshape b/c we will convert to np.array later...
            episode_actions += [action[0]]
            total_reward += reward
            if done:
                break
        return episode_states, episode_actions, [total_reward] * len(episode_actions)

    # run test
    with tf.Session() as sess:
        saver.restore(sess, find_latest_checkpoint(restore_dir))
        rewards = []
        for i in range(episodes):
            _, _, total_rewards = run_episode(env, sess, render=render)
            rewards += [np.mean(total_rewards)]
        return rewards

if __name__ == '__main__':
    hidden_units = [int(i) for i in FLAGS.hidden_units.split(',')]
    if FLAGS.mode == 'train':
        checkpoint_file = train(env_name=FLAGS.env_name,
                                device=FLAGS.device,
                                hidden_units=hidden_units,
                                learning_rate=FLAGS.learning_rate,
                                batches=FLAGS.batches,
                                batch_size=FLAGS.batch_size,
                                ckpt_freq=FLAGS.ckpt_freq,
                                base_dir=FLAGS.base_dir,
                                render=FLAGS.render)
        print('Checkpoint saved to {}'.format(checkpoint_file))
    elif FLAGS.mode == 'test':
        rewards = test(env_name=FLAGS.env_name,
                       hidden_units=hidden_units,
                       episodes=FLAGS.episodes,
                       load_path=FLAGS.load_path,
                       render=FLAGS.render)
        print("> avg. reward = {:.2f} ({:.2f})".format(np.mean(rewards), np.std(rewards)))
