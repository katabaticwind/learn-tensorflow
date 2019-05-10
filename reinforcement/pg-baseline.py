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
tf.app.flags.DEFINE_boolean('atari', False, """Flag for Atari environments.""")
tf.app.flags.DEFINE_string('env_name', 'CartPole-v0', """Gym environment.""")
tf.app.flags.DEFINE_string('filters', '16,32', """Number of filters in CNN layers.""")
tf.app.flags.DEFINE_string('hidden_units', '32', """Size of hidden layers.""")
tf.app.flags.DEFINE_float('learning_rate', '1e-2', """Initial learning rate.""")
tf.app.flags.DEFINE_integer('batches', 100, """Batches per training update.""")
tf.app.flags.DEFINE_integer('episodes', 100, """Episodes per test.""")
tf.app.flags.DEFINE_string('save_path', './checkpoints/', """Checkpoint directory.""")
tf.app.flags.DEFINE_string('load_path', './checkpoints/', """Checkpoint directory.""")
tf.app.flags.DEFINE_boolean('render', False, """Render once per batch in training mode.""")


# TODO: re-write mlp with low-level API (tf.Variables)
# TODO: logits for action selection on CPU; logits for updates on GPU? (During batch construction, shape of logits is (1, 4); during update step shape is (5000, 4)).

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

def cnn(x, filters, units, activation=tf.nn.relu, output_activation=None):

    x = tf.reshape(x, (-1, 84, 84, 4))

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

def train(env_name='CartPole-v0', hidden_units=[32], learning_rate=1e-2, batches=100, batch_size=5000, save_path=None, render=False):

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

    # define core functions (NOTE: you could also define w/in session scope and drop sess arg)
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
        batch_rewards = []
        batch_steps = []
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
            batch_rewards.append(total_reward)
            batch_steps.append(total_steps)
        t = time.time()
        return batch_states, batch_actions, batch_weights, np.mean(batch_rewards), np.mean(batch_steps), episodes, t - t0

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
        for batch in range(batches):
            batch_states, batch_actions, batch_weights, batch_reward, batch_steps, batch_episodes, batch_time = run_batch(env, sess, render)
            _ = update_value_network(batch_states, batch_weights, sess)
            _ = update_policy_network(batch_states, batch_actions, batch_weights, sess)
            print("batch = {:d},    reward = {:.2f} (mean),    steps = {:.2f} (mean),   time = {:.2f}".format(batch, batch_reward, batch_steps, batch_time))
        if save_path is not None:
            saver.save(sess, save_path=save_path + 'vpg-baseline-' + env_name)
            return saver.last_checkpoints

def train_atari(env_name='Breakout-v0', filters=[16, 32], hidden_units=[1024], learning_rate=1e-6, episodes=100, batch_size=32, save_path=None, render=False):

    # create an environment
    env = gym.make(env_name)
    n_actions = available_actions(env)

    # create placeholders
    states_pl = tf.placeholder(tf.float32, (None, 84, 84, 4))  # 4 frames per state
    actions_pl = tf.placeholder(tf.int32, (None, ))
    weights_pl = tf.placeholder(tf.float32, (None, ))

    # create a policy network
    logits = cnn(states_pl, filters, hidden_units + [n_actions])
    actions = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)

    # create a value network
    values = cnn(states_pl, filters, hidden_units + [1])

    # define policy network training operation
    actions_mask = tf.one_hot(actions_pl, n_actions)
    log_probs = tf.reduce_sum(actions_mask * tf.nn.log_softmax(logits), axis=1)
    policy_loss = -tf.reduce_mean((weights_pl - values) * log_probs)
    policy_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(policy_loss)

    # define value network training operation
    value_loss = tf.losses.mean_squared_error(weights_pl, tf.squeeze(values, axis=1))
    value_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(value_loss)

    # create a saver
    saver = tf.train.Saver()

    # core functions
    def run_episode(env, sess, render=False):
        frame = env.reset()
        frame_queue = deque()
        append_to_queue(frame_queue, frame, max_len=4)
        state = convert_frames(frame_queue, min_len=4)
        episode_states = [state]
        episode_actions = []
        episode_weights = []
        total_reward = 0
        total_steps = 0
        if render == True:
            env.render()
        while True:
            action = sess.run(actions, feed_dict={states_pl: state.reshape(1, 84, 84, 4)})[0]
            frame, reward, done, info = env.step(action)  # step requires scalar action
            if render == True:
                env.render()
            append_to_queue(frame_queue, frame, max_len=4)
            state = convert_frames(frame_queue, min_len=4)
            episode_states += [state]  # copy needed?
            episode_actions += [action]
            episode_weights += [reward]
            total_reward += reward
            total_steps += 1
            # print("steps={}, action={}".format(total_steps, action))
            if done:
                break
        return episode_states[:-1], episode_actions, reward_to_go(episode_weights), total_reward, total_steps

    def update_policy_network(states, actions, weights, sess):
        idx = 0
        while idx < len(states):
            feed_dict = {
                states_pl: states[idx:idx + batch_size],
                weights_pl: weights[idx:idx + batch_size],
                actions_pl: actions[idx:idx + batch_size]
            }
            batch_loss, _ = sess.run([policy_loss, policy_train_op], feed_dict=feed_dict)
            idx += batch_size

    def update_value_network(states, weights, sess):
        idx = 0
        while idx < len(states):
            feed_dict = {
                states_pl: states[idx:idx + batch_size],
                weights_pl: weights[idx:idx + batch_size],
            }
            batch_loss, _ = sess.run([value_loss, value_train_op], feed_dict=feed_dict)
            idx += batch_size

    # train the network
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t0 = time.time()
        avg_reward = 0
        for episode in range(episodes):
            if (episode + 1) % 10 == 0:
                batch_states, batch_actions, batch_weights, reward, steps = run_episode(env, sess, True)
            else:
                batch_states, batch_actions, batch_weights, reward, steps = run_episode(env, sess, render)
            avg_reward += reward
            update_value_network(batch_states, batch_weights, sess)
            update_policy_network(batch_states, batch_actions, batch_weights, sess)
            if (episode + 1) % 10 == 0:
                elapsed_time = time.time() - t0
                avg_reward = avg_reward / 10
                print("episode: {:d},  reward: {:.2f},  elapsed_time: {:.2f}".format(episode + 1, avg_reward, elapsed_time))
                avg_reward = 0
                if save_path is not None:
                    saver.save(sess, save_path=save_path + 'pg-baseline-' + env_name, global_step=episode + 1)
        return saver.last_checkpoints

def test(env_name='CartPole-v0', hidden_units=[32], episodes=100, load_path=None, render=False):
    """
    Load and test a trained model from checkpoint files.

    **Note**: the `load_path` is the part of checkpoint file name *before* the extension. For example, if the checkpoint file name is 'model.index', then use 'model'.
    """

    if load_path is not None:
        print('\nTesting environment {} with agent found in {}...'.format(env_name, load_path + 'vpg-baseline-' + env_name))

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
    def run_episode(env, sess, render=False):
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
        saver.restore(sess, load_path + 'vpg-baseline-' + env_name)
        rewards = []
        for i in range(episodes):
            _, _, total_rewards = run_episode(env, sess, render=render)
            rewards += [np.mean(total_rewards)]
        return rewards

def test_atari(env_name='Breakout-v0', filters=[32, 64], hidden_units=[1024], load_path=None, render=False):
    pass

if __name__ == '__main__':
    filters = [int(i) for i in FLAGS.filters.split(',')]
    hidden_units = [int(i) for i in FLAGS.hidden_units.split(',')]
    if FLAGS.mode == 'train':
        if FLAGS.atari:
            checkpoint_file = train_atari(env_name=FLAGS.env_name,
                                          filters=filters,
                                          hidden_units=hidden_units,
                                          learning_rate=FLAGS.learning_rate,
                                          episodes=FLAGS.batches,
                                          save_path=FLAGS.save_path,
                                          render=FLAGS.render)
        else:
            checkpoint_file = train(env_name=FLAGS.env_name,
                                    hidden_units=hidden_units,
                                    learning_rate=FLAGS.learning_rate,
                                    batches=FLAGS.batches,
                                    save_path=FLAGS.save_path,
                                    render=FLAGS.render)
        print('Checkpoint saved to {}'.format(checkpoint_file))
    elif FLAGS.mode == 'test':
        if FLAGS.atari:
            rewards = test_atari(env_name=FLAGS.env_name,
                                 episodes=FLAGS.episodes,
                                 load_path=FLAGS.load_path,
                                 render=FLAGS.render)
        else:
            rewards = test(env_name=FLAGS.env_name,
                           episodes=FLAGS.episodes,
                           load_path=FLAGS.load_path,
                           render=FLAGS.render)
        print("> mean = {:.2f}\n> std = {:.2f}".format(np.mean(rewards), np.std(rewards)))
