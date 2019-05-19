import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all messages
import numpy as np
import gym
import time
from collections import deque

from atari import crop_frame, RGB_to_luminance, convert_frames, append_to_queue

tf.logging.set_verbosity(tf.logging.ERROR)  # suppress annoying messages


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'train', """'Train' or 'test'.""")
tf.app.flags.DEFINE_string('network', 'mlp', """'mlp' or 'cnn'""")
tf.app.flags.DEFINE_string('env_name', 'Pong-v0', """Gym environment.""")
tf.app.flags.DEFINE_string('filters', '8,16', """Number of filters in CNN layers.""")
tf.app.flags.DEFINE_string('hidden_units', '256,128', """Size of hidden layers.""")
tf.app.flags.DEFINE_float('lr', '1e-3', """Initial learning rate.""")
tf.app.flags.DEFINE_integer('batches', 1000, """Batches per training run.""")
tf.app.flags.DEFINE_integer('episodes_per_batch', 10, """Episodes per batch.""")
tf.app.flags.DEFINE_integer('frames_per_state', 2, """Frames used to construct a state.""")
tf.app.flags.DEFINE_integer('report_freq', 1, """Batches between reports.""")
tf.app.flags.DEFINE_string('save_path', './checkpoints/', """Checkpoint directory.""")
tf.app.flags.DEFINE_string('load_path', './checkpoints/', """Checkpoint directory.""")
tf.app.flags.DEFINE_boolean('render', False, """Render once per batch in training mode.""")


def available_actions(env):
    try:
        return env.action_space.n
    except AttributeError:
        raise AttributeError("env.action_space is not Discrete")

def state_dimensions(env):
    """Find the number of dimensions in the state."""
    return env.observation_space.shape[0]

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    """Build a feedforward neural network."""
    x = tf.reshape(x, (-1, 84 * 84 * x.shape[-1].value))  # x.shape[-1].value gives last dim size as int
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)  # TODO: creates multiple warnings (DEPRECATED)
    return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def cnn(x, filters, units, activation=tf.nn.relu, output_activation=None):

    # conv1
    x = tf.layers.conv2d(
        inputs=x,
        filters=filters[0],
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=2
    )  # batch_size x 42 x 42 x filters[0]

    # conv2
    x = tf.layers.conv2d(
        inputs=x,
        filters=filters[1],
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=2
    )  # batch_size x 21 x 21 x filters[1]

    # dense
    x = tf.reshape(x, (-1, 21 * 21 * filters[1]))  # flatten x
    x = tf.layers.dense(x, units[0], tf.nn.relu)  # 1024

    # logits
    return tf.layers.dense(x, units[-1], activation=output_activation)

def reward_to_go(rewards):
    """Calculate the cumulative reward at each step."""
    c = []
    for (i, r) in enumerate(reversed(rewards)):
        if i == 0:
            c += [r]
        else:
            c += [r + c[i - 1]]
    return list(reversed(c))

def train(env_name='Pong-v0', network='mlp', filters=[8, 16], hidden_units=[256, 128], lr=1e-3, batches=1000, episodes_per_batch=5, frames_per_state=2, report_freq=1, save_path=None, render=False):

    # create an environment
    env = gym.make(env_name)
    n_actions = available_actions(env)

    # create placeholders
    states_pl = tf.placeholder(tf.float32, (None, 84, 84, frames_per_state))
    actions_pl = tf.placeholder(tf.int32, (None, ))
    weights_pl = tf.placeholder(tf.float32, (None, ))

    # create a policy network
    if network == 'mlp':
        logits = mlp(states_pl, hidden_units + [n_actions])
    elif network == 'cnn':
        logits = cnn(states_pl, filters, hidden_units + [n_actions])
    actions = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)

    # create a value network
    if network == 'mlp':
        values = mlp(states_pl, hidden_units + [1])
    elif network == 'cnn':
        values = cnn(states_pl, filters, hidden_units + [1])

    # define policy network training operation
    actions_mask = tf.one_hot(actions_pl, n_actions)
    log_probs = tf.reduce_sum(actions_mask * tf.nn.log_softmax(logits), axis=1)
    policy_loss = -tf.reduce_mean((weights_pl - values) * log_probs)
    policy_train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(policy_loss)

    # define value network training operation
    value_loss = tf.losses.mean_squared_error(weights_pl, tf.squeeze(values, axis=1))
    value_train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(value_loss)

    # create a saver
    saver = tf.train.Saver()

    # core functions
    def run_episode(env, sess, render=False):
        frame = env.reset()
        frame_queue = deque()
        append_to_queue(frame_queue, frame, max_len=frames_per_state)
        state = convert_frames(frame_queue, min_len=frames_per_state)
        episode_states = [state]
        episode_actions = []
        episode_weights = []
        total_reward = 0
        total_steps = 0
        if render == True:
            env.render()
        while True:
            action = sess.run(actions, feed_dict={states_pl: state.reshape(1, 84, 84, frames_per_state)})[0]
            frame, reward, done, info = env.step(action)  # step requires scalar action
            if render == True:
                env.render()
            append_to_queue(frame_queue, frame, max_len=frames_per_state)
            state = convert_frames(frame_queue, min_len=frames_per_state)
            episode_states += [state]
            episode_actions += [action]
            episode_weights += [reward]
            total_reward += reward
            total_steps += 1
            if done:
                break
        return episode_states[:-1], episode_actions, reward_to_go(episode_weights), total_reward, total_steps

    def run_batch(env, sess, render=False):
        t0 = time.time()
        batch_states = []
        batch_actions = []
        batch_weights = []
        batch_rewards = 0
        batch_steps = 0
        episodes = 0
        while episodes < episodes_per_batch:
            if episodes == 0:
                episode_states, episode_actions, episode_weights, total_reward, total_steps = run_episode(env, sess, render=render)  # (optionally) render first episode per batch
            else:
                episode_states, episode_actions, episode_weights, total_reward, total_steps = run_episode(env, sess, render=False)
            batch_states.extend(episode_states)
            batch_actions.extend(episode_actions)
            batch_weights.extend(episode_weights)
            batch_rewards += total_reward
            batch_steps += total_steps
            episodes += 1
        elapsed_time = time.time() - t0
        return batch_states, batch_actions, batch_weights, batch_rewards / episodes_per_batch, batch_steps, elapsed_time

    def update_policy_network(states, actions, weights, sess):
        feed_dict = {
            states_pl: states,
            weights_pl: weights,
            actions_pl: actions
        }
        batch_loss, _ = sess.run([value_loss, value_train_op], feed_dict=feed_dict)

    def update_value_network(states, weights, sess):
        feed_dict = {
            states_pl: states,
            weights_pl: weights
        }
        batch_loss, _ = sess.run([value_loss, value_train_op], feed_dict=feed_dict)

    # train the network
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for batch in range(batches):
            batch_states, batch_actions, batch_weights, reward, steps, elapsed_time = run_batch(env, sess, render=render)
            update_value_network(batch_states, batch_weights, sess)
            update_policy_network(batch_states, batch_actions, batch_weights, sess)
            print("batch: {:d},  (avg) reward: {:.2f},  (total) steps: {:.2f},  elapsed_time: {:.2f}".format(batch + 1, reward, steps, elapsed_time))
            if (save_path is not None) and (batch + 1 % report_freq == 0):
                saver.save(sess, save_path=save_path + 'pg-atari-' + env_name, global_step=batch + 1)
        return saver.last_checkpoints

def test(env_name='Pong-v0', filters=[8, 16], hidden_units=[512], episodes=100, load_path=None, render=False):
    pass

if __name__ == '__main__':
    if FLAGS.mode == 'train':
        checkpoint_file = train(env_name=FLAGS.env_name,
                                network=FLAGS.network,
                                filters=[int(i) for i in FLAGS.filters.split(',')],
                                hidden_units=[int(i) for i in FLAGS.hidden_units.split(',')],
                                lr=FLAGS.lr,
                                batches=FLAGS.batches,
                                episodes_per_batch=FLAGS.episodes_per_batch,
                                frames_per_state=FLAGS.frames_per_state,
                                report_freq=FLAGS.report_freq,
                                save_path=FLAGS.save_path,
                                render=FLAGS.render)
        print('Checkpoint saved to {}'.format(checkpoint_file))
    elif FLAGS.mode == 'test':
        rewards = test(env_name=FLAGS.env_name,
                       filters=[int(i) for i in FLAGS.filters.split(',')],
                       hidden_units=[int(i) for i in FLAGS.hidden_units.split(',')],
                       episodes=FLAGS.episodes,
                       load_path=FLAGS.load_path,
                       render=FLAGS.render)
        print("> mean = {:.2f}\n> std = {:.2f}".format(np.mean(rewards), np.std(rewards)))
