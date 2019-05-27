import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all messages
import numpy as np
import gym
import time

from utils import create_directories
from priority_queue import Queue, PriorityQueue, unpack_batch

tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', 'train', """'Train' or 'test'.""")
tf.app.flags.DEFINE_string('device', '/cpu:0', """'/cpu:0' or '/device:GPU:0'.""")
tf.app.flags.DEFINE_string('env_name', 'CartPole-v0', """Gym environment.""")
tf.app.flags.DEFINE_string('hidden_units', '64', """Size of hidden layers.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """Initial learning rate.""")
tf.app.flags.DEFINE_float('lr_decay', 1.0, """Learning decay rate (per episode).""")
tf.app.flags.DEFINE_float('discount_factor', 0.99, """Discount factor in update.""")
tf.app.flags.DEFINE_float('init_epsilon', 1.0, """Initial exploration rate.""")
tf.app.flags.DEFINE_float('min_epsilon', 0.01, """Minimum exploration rate.""")
tf.app.flags.DEFINE_float('eps_decay', 0.995, """Exploration parameter decay rate (per episode).""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Examples per training update.""")
tf.app.flags.DEFINE_integer('episodes', 10000, """Episodes per train/test.""")
tf.app.flags.DEFINE_integer('update_freq', 4, """Actions/steps between updates.""")
tf.app.flags.DEFINE_integer('clone_steps', 10000, """Steps between cloning ops.""")
tf.app.flags.DEFINE_integer('max_steps', 1000, """Maximum steps per episode.""")
tf.app.flags.DEFINE_integer('memory_size', 100000, """Maximum number of replay memories.""")
tf.app.flags.DEFINE_float('init_alpha', 0.5, """Initial memory replay alpha.""")  # anneal to 0.0
tf.app.flags.DEFINE_float('init_beta', 0.5, """Initial memory replay beta.""")  # anneal to 1.0 (`1 - beta` to 0.0)
tf.app.flags.DEFINE_float('memory_decay', 0.99, """Memory parameter annealing rate (per episode).""")
tf.app.flags.DEFINE_integer('sort_freq', 1000000, """Steps between memory replay sorts.""")
tf.app.flags.DEFINE_integer('ckpt_freq', 100, """Steps per checkpoint.""")
tf.app.flags.DEFINE_string('base_dir', '.', """Base directory for checkpoints and logs.""")
tf.app.flags.DEFINE_boolean('render', False, """Render episodes (once per `ckpt_freq` in training mode).""")

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

def log_scalar(writer, tag, value, step):
    value = [tf.Summary.Value(tag=tag, simple_value=value)]
    summary = tf.Summary(value=value)
    writer.add_summary(summary, step)

def discounted_sum(values, discount_factor):
    return np.sum(values * discount_factor ** np.arange(0, len(values)))

def train(env_name='CartPole-v0',
          device='/cpu:0',
          hidden_units=[64,64],
          learning_rate=1e-3,
          lr_decay=0.995,
          discount_factor=0.99,
          init_epsilon=1.0,
          min_epsilon=0.01,
          eps_decay=0.995,
          episodes=1000,
          update_freq=4,
          batch_size=32,
          clone_steps=1000,
          max_steps=1000,
          memory_size=100000,
          replay_alpha=0.5,
          replay_beta=1.0,
          memory_decay=0.99,
          sort_freq=1000000,
          ckpt_freq=100,
          base_dir=None,  # "."
          render=True):

    # create log and checkpoint directories
    if base_dir is not None:
        ckpt_dir, log_dir = create_directories(env_name, "dqn-prioritized", base_dir=base_dir)
    else:
        ckpt_dir = log_dir = None

    # create an environment
    env = gym.make(env_name)
    n_dims = state_dimensions(env)
    n_actions = available_actions(env)

    # define a graph
    with tf.device(device):

        print('constructing graph on device: {}'.format(device))

        # create placeholders
        states_pl = tf.placeholder(tf.float32, (None, n_dims))
        actions_pl = tf.placeholder(tf.int32, (None,))
        targets_pl = tf.placeholder(tf.float32, (None,))
        weights_pl = tf.placeholder(tf.float32, (None,))  # importance-sampling weights

        # initialize networks
        action_values = mlp(states_pl, hidden_units + [n_actions], scope='value')
        target_values = mlp(states_pl, hidden_units + [n_actions], scope='target')
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
        replay_memory = PriorityQueue(env, memory_size, replay_alpha, replay_beta, batch_size)

        # define training operation
        # errors = tf.math.abs(targets_pl - values)
        errors = tf.abs(targets_pl - values)
        loss = tf.losses.mean_squared_error(values, targets_pl, weights=weights_pl)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # create a saver
        saver = tf.train.Saver()

        # create writer
        writer = tf.summary.FileWriter(log_dir)

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

    def run_episode(env, sess, global_step, global_epsilon, render=False, delay=0.0):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        reward_queue = Queue(update_freq)
        action_queue = Queue(update_freq)
        state_queue = Queue(update_freq + 1, [state])
        if render == True:
            env.render()
            time.sleep(delay)
        while True:
            # select an action
            if np.random.rand() < global_epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = sess.run(greedy_action, feed_dict={states_pl: state.reshape(1, -1)})[0]
            action_queue.push(action)

            # take a step
            next_state, reward, done, info = env.step(action)
            if render == True:
                env.render()
                time.sleep(delay)

            # update queues
            reward_queue.push(reward)
            state_queue.push(next_state)

            # save transition to memory
            if len(state_queue) == update_freq + 1:
                memory = [state_queue.queue[0],
                          action_queue.queue[0],
                          discounted_sum(reward_queue.queue, discount_factor),
                          state_queue.queue[-1],
                          done]
                replay_memory.add_memory(memory, global_step)  # stored w/ maximal priority
                state = next_state  # .copy() is only necessary if you modify the contents of next_state

            # update episode totals
            episode_reward += reward
            episode_steps += 1

            # perform update
            if episode_steps % update_freq == 0:
                # sample batch
                batch_idx, batch = replay_memory.sample_queue()
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = unpack_batch(batch)
                # find importance-sampling weights
                batch_weights = [replay_memory.calculate_weight(i + 1) for i in batch_idx]
                # find targets
                batch_targets = sess.run(targets,
                    feed_dict={
                        states_pl: batch_next_states,
                    })
                batch_targets = batch_rewards + ~batch_dones * discount_factor * batch_targets
                # parameter update *and* td-error calcuation
                td_errors, _ = sess.run([errors, train_op],
                    feed_dict={
                        states_pl: batch_states,
                        actions_pl: batch_actions,
                        targets_pl: batch_targets,
                        weights_pl: batch_weights
                    })
                # update memory priorities
                [replay_memory.update_memory(i, -e) for (i, e) in zip(batch_idx, td_errors)]

            # update global step count
            global_step += 1

            # update target network
            if global_step % clone_steps == 0:
                sess.run(clone_ops)
                # print("updated target network")

            # sort replay memory
            if global_step % sort_freq == 0:
                replay_memory.sort_queue()


            # check if episode is done
            if done:
                global_epsilon = np.maximum(min_epsilon, global_epsilon * eps_decay)
                # print("  > reward: {:.2f}, steps: {:d}".format(episode_reward, episode_steps))
                break

            if episode_steps == max_steps:
                global_epsilon = np.maximum(min_epsilon, global_epsilon * eps_decay)
                print("reached max_steps")
                break

        return episode_reward, episode_steps, global_step, global_epsilon

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(clone_ops)  # set networks equal to begin
        global_step = 0
        global_epsilon = init_epsilon
        t0 = time.time()
        reward_history = []
        for episode in range(episodes):
            if (episode + 1) % ckpt_freq == 0:
                reward, steps, global_step, global_epsilon = run_episode(env, sess, global_step, global_epsilon, render=render)  # optionally render one episode per checkpoint
            else:
                reward, steps, global_step, global_epsilon = run_episode(env, sess, global_step, global_epsilon, render=False)
            learning_rate = learning_rate * lr_decay
            replay_alpha = memory_decay * replay_alpha
            replay_beta = 1 - memory_decay * (1 - replay_beta)
            replay_memory.set_alpha(replay_alpha)
            replay_memory.set_beta(replay_beta)
            reward_history += [reward]
            elapsed_time = time.time() - t0
            avg_reward = np.mean(reward_history[-100:])
            print('episode: {:d},  reward: {:.2f},  avg. reward: {:.2f},  steps:  {:d},  epsilon: {:.2f}, lr: {:.2e},  elapsed: {:.2f}'.format(episode + 1, reward, avg_reward, global_step, global_epsilon, learning_rate, elapsed_time))

            # logging
            log_scalar(writer, 'reward', reward, global_step)
            log_scalar(writer, 'learning_rate', learning_rate, global_step)
            log_scalar(writer, 'epsilon', global_epsilon, global_step)
            log_scalar(writer, 'replay_alpha', replay_alpha, global_step)
            log_scalar(writer, 'replay_beta', replay_beta, global_step)

            # checkpoint
            if (episode + 1) % ckpt_freq == 0:
                if ckpt_dir is not None:
                    saver.save(sess, save_path=ckpt_dir + "/ckpt", global_step=global_step)

        # final checkpoint
        if ckpt_dir is not None:
            saver.save(sess, save_path=ckpt_dir + "/ckpt", global_step=global_step)

def find_latest_checkpoint(ckpt_dir, prefix="ckpt-"):
    """Find the latest checkpoint in dir at `load_path` with prefix `prefix`

        E.g. ./checkpoints/CartPole-v0/dqn-prioritized/GLOBAL_STEP
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
              hidden_units=[int(i) for i in FLAGS.hidden_units.split(',')],
              learning_rate=FLAGS.learning_rate,
              lr_decay=FLAGS.lr_decay,
              discount_factor=FLAGS.discount_factor,
              init_epsilon=FLAGS.init_epsilon,
              min_epsilon=FLAGS.min_epsilon,
              eps_decay=FLAGS.eps_decay,
              episodes=FLAGS.episodes,
              update_freq=FLAGS.update_freq,
              batch_size=FLAGS.batch_size,
              clone_steps=FLAGS.clone_steps,
              max_steps=FLAGS.max_steps,
              ckpt_freq=FLAGS.ckpt_freq,
              memory_size=FLAGS.memory_size,
              replay_alpha=FLAGS.init_alpha,
              replay_beta=FLAGS.init_beta,
              memory_decay=FLAGS.memory_decay,
              sort_freq=FLAGS.sort_freq,
              base_dir=FLAGS.base_dir,
              render=FLAGS.render)
    else:
        test(env_name=FLAGS.env_name,
             hidden_units=[int(i) for i in FLAGS.hidden_units.split(',')],
             epsilon=FLAGS.min_epsilon,
             episodes=FLAGS.episodes,
             restore_dir=FLAGS.base_dir,
             render=FLAGS.render)
