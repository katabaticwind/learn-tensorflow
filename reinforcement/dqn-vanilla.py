import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all messages
import numpy as np
import gym
import time
from collections import deque  # for replay memory


tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', 'train', """'Train' or 'test'.""")
tf.app.flags.DEFINE_string('env_name', 'CartPole-v0', """Gym environment.""")
tf.app.flags.DEFINE_string('hidden_units', '64,32', """Size of hidden layers.""")
tf.app.flags.DEFINE_float('lr', '1e-3', """Initial learning rate.""")
tf.app.flags.DEFINE_float('init_epsilon', 1.0, """Initial exploration rate.""")
tf.app.flags.DEFINE_float('min_epsilon', 0.01, """Minimum exploration rate.""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Examples per training update.""")
tf.app.flags.DEFINE_integer('episodes', 1000, """Episodes per train/test routine.""")
tf.app.flags.DEFINE_integer('clone_steps', 1000, """Steps between cloning ops.""")
tf.app.flags.DEFINE_integer('anneal_steps', 10000, """Steps per train/test routine.""")
tf.app.flags.DEFINE_integer('min_memory_size', 10000, """Minimum number of replay memories.""")
tf.app.flags.DEFINE_integer('max_memory_size', 100000, """Maximum number of replay memories.""")
tf.app.flags.DEFINE_integer('checkpoint_freq', 25, """Steps per checkpoint.""")
tf.app.flags.DEFINE_string('save_path', './checkpoints/', """Checkpoint directory.""")
tf.app.flags.DEFINE_string('load_path', './checkpoints/', """Checkpoint directory.""")
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

def mlp(x, sizes, activation=tf.tanh, output_activation=None, scope=''):
    """Build a feedforward neural network.

        - `scope`: used to distinguish target network from value network.

    """
    with tf.variable_scope(scope):
        for size in sizes[:-1]:
            x = tf.layers.dense(x, units=size, activation=activation)  # TODO: creates multiple warnings (DEPRECATED)
        return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def create_replay_memory(env, min_memory_size, max_memory_size):
    """Initialize replay memory to `size`. Collect experience under random policy."""

    print('Creating replay memory...')
    t0 = time.time()
    memory = deque()
    while len(memory) < min_memory_size:
        state = env.reset()
        while True:
            action = np.random.randint(env.action_space.n)
            next_state, reward, done, info = env.step(action)
            update_memory(memory, [state, action, next_state, reward, done], max_memory_size)
            state = next_state
            if done or len(memory) == min_memory_size:
                break
    elapsed_time = time.time() - t0
    print('done (elapsed time: {:.2f})'.format(elapsed_time))
    return memory

def update_memory(memory, transition, size):
    """Add a transition to `memory` with maximum size `size`."""
    if len(memory) == size:
        memory.popleft()  # trash oldest memory
        memory.append(transition)
    else:
        memory.append(transition)

def sample_memory(memory, size):
    """Sample `size` transitions from `memory` uniformly"""
    idx = np.random.choice(range(len(memory)), size)
    batch = [memory[i] for i in idx]
    states = np.array([b[0] for b in batch])
    actions = np.array([b[1] for b in batch])
    next_states = np.array([b[2] for b in batch])
    rewards = np.array([b[3] for b in batch])
    dones = np.array([b[4] for b in batch])
    return states, actions, next_states, rewards, dones

def train(env_name='CartPole-v0', hidden_units=[64,32], lr=1e-3, init_epsilon=1.0, min_epsilon=0.01, batch_size=32, episodes=1000, clone_steps=1000, anneal_steps=10000, min_memory_size=10000, max_memory_size=100000, checkpoint_freq=25, save_path=None, render=False):

    # create an environment
    env = gym.make(env_name)
    n_dims = state_dimensions(env)
    n_actions = available_actions(env)

    # create placeholders
    states_pl = tf.placeholder(tf.float32, (None, n_dims))
    actions_pl = tf.placeholder(tf.int32, (None, ))
    targets_pl = tf.placeholder(tf.float32, (None, ))

    # initialize networks
    action_values = mlp(states_pl, hidden_units + [n_actions], scope='value')
    target_values = mlp(states_pl, hidden_units + [n_actions], scope='target')
    # greedy_action = tf.math.argmax(action_values, axis=1)
    # target_actions = tf.math.argmax(target_values, axis=1)
    greedy_action = tf.arg_max(action_values, dimension=1)
    target_actions = tf.arg_max(target_values, dimension=1)
    value_mask = tf.one_hot(actions_pl, n_actions)
    target_mask = tf.one_hot(target_actions, n_actions)
    values = tf.reduce_sum(value_mask * action_values, axis=1)
    targets = tf.reduce_sum(target_mask * target_values, axis=1)  # minus reward

    # define cloning operation
    source = tf.get_default_graph().get_collection('trainable_variables', scope='value')
    target = tf.get_default_graph().get_collection('trainable_variables', scope='target')
    clone_ops = [tf.assign(t, s, name='clone') for t,s in zip(target, source)]

    # initialize replay memory
    memory = create_replay_memory(env, min_memory_size, max_memory_size)

    # define training operation
    loss = tf.losses.mean_squared_error(values, targets_pl)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # create a saver
    saver = tf.train.Saver()

    def clone_network(sess):
        """Clone `action_values` network to `target_values`."""
        sess.run(clone_ops)

    def anneal_epsilon(step, init_epsilon=1.0, min_epsilon=0.1, anneal_steps=20000):
        """
            Linear annealing of `init_epsilon` to `min_epsilon` over `anneal_steps`.

            (Default `init_epsilon` and `min_epsilon` are the same as Mnih et al. 2015).
        """

        epsilon = init_epsilon - step * (init_epsilon - min_epsilon) / anneal_steps
        return np.maximum(min_epsilon, epsilon)

    def run_episode(env, sess, global_step, global_epsilon, render=False, delay=0.01):
        state = env.reset()
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
                action = sess.run(greedy_action, feed_dict={states_pl: state.reshape(1, -1)})[0]

            # take a step
            next_state, reward, done, info = env.step(action)
            if render == True:
                env.render()
                time.sleep(delay)

            # save transition to memory
            update_memory(memory, [state, action, next_state, reward, done], max_memory_size)
            state = next_state  # .copy() is only necessary if you modify the contents of next_state

            # perform update
            batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = sample_memory(memory, batch_size)
            batch_targets = sess.run(targets,
                feed_dict={
                    states_pl: batch_next_states,
                })
            batch_targets = batch_rewards + ~batch_dones * batch_targets
            sess.run([loss, train_op],
                feed_dict={
                    states_pl: batch_states,
                    actions_pl: batch_actions,
                    targets_pl: batch_targets
                })

            # update step count and clone
            global_step += 1
            if global_step % clone_steps == 0:
                sess.run(clone_ops)
                # print("Cloned action-value network!")

            # update epsilon
            global_epsilon = anneal_epsilon(global_step, init_epsilon,
                                            min_epsilon, anneal_steps)

            # update episode totals
            episode_reward += reward
            episode_steps += 1

            # check if you're done
            if done:
                break

        return episode_reward, episode_steps, global_step, global_epsilon

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(clone_ops)  # set networks equal to begin
        global_step = 0
        global_epsilon = init_epsilon
        t0 = time.time()
        total_reward = 0
        for episode in range(episodes):
            if (episode + 1) % checkpoint_freq == 0:
                episode_reward, episode_steps, global_step, global_epsilon = run_episode(env, sess, global_step, global_epsilon, render=render)
            else:
                episode_reward, episode_steps, global_step, global_epsilon = run_episode(env, sess, global_step, global_epsilon, render=False)
            total_reward += episode_reward
            if (episode + 1) % checkpoint_freq == 0:
                elapsed_time = time.time() - t0
                mean_reward = total_reward / checkpoint_freq
                print('episode: {:d},  reward: {:.2f},  (global) steps:  {:d},  (global) epsilon: {:.2f},  elapsed: {:.2f}'.format(episode + 1, mean_reward, global_step, global_epsilon, elapsed_time))
                if save_path is not None:
                    saver.save(sess, save_path=save_path + 'dqn-vanilla-' + env_name, global_step=global_step)
                total_reward = 0
        if save_path is not None:
            saver.save(sess, save_path=save_path + 'dqn-vanilla-' + env_name, global_step=global_step)
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

def test(env_name='CartPole-v0', hidden_units=[32], epsilon=0.01, episodes=100, load_path=None, render=False):

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
        prefix = 'dqn-vanilla-' + env_name
        saver.restore(sess, find_latest_checkpoint(load_path, prefix))
        rewards = 0
        for i in range(episodes):
            episode_rewards, _ = run_episode(env, sess, epsilon, render=render)
            rewards += episode_rewards
        return rewards / episodes

if __name__ == '__main__':
    hidden_units = [int(i) for i in FLAGS.hidden_units.split(',')]
    if FLAGS.mode == 'train':
        train(env_name=FLAGS.env_name,
              hidden_units=hidden_units,
              lr=FLAGS.lr,
              init_epsilon=FLAGS.init_epsilon,
              min_epsilon=FLAGS.min_epsilon,
              batch_size=FLAGS.batch_size,
              episodes=FLAGS.episodes,
              clone_steps=FLAGS.clone_steps,
              anneal_steps=FLAGS.anneal_steps,
              min_memory_size=FLAGS.min_memory_size,
              max_memory_size=FLAGS.max_memory_size,
              save_path=FLAGS.save_path,
              render=FLAGS.render)
    else:
        score = test(env_name=FLAGS.env_name,
                     hidden_units=hidden_units,
                     epsilon=FLAGS.min_epsilon,
                     episodes=FLAGS.episodes,
                     load_path=FLAGS.load_path,
                     render=FLAGS.render)
        print("> mean = {:.2f}".format(score))
