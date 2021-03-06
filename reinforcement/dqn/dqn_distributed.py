import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all messages
import numpy as np
import gym
import time
from collections import deque  # for replay memory

from utils import create_directories

tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', 'train', """'Train' or 'test'.""")
tf.app.flags.DEFINE_string('device', '/cpu:0', """'/cpu:0' or '/device:GPU:0'.""")
tf.app.flags.DEFINE_string('env_name', 'CartPole-v0', """Gym environment.""")
tf.app.flags.DEFINE_string('hidden_units', '64', """Size of hidden layers.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """Initial learning rate.""")
tf.app.flags.DEFINE_float('lr_decay', 1.0, """Learning decay rate (per episode).""")
tf.app.flags.DEFINE_float('discount_factor', 0.99, """Discount factor in update (gamma).""")
tf.app.flags.DEFINE_float('init_epsilon', 1.0, """Initial exploration rate.""")
tf.app.flags.DEFINE_float('min_epsilon', 0.01, """Minimum exploration rate.""")
tf.app.flags.DEFINE_float('eps_decay', 0.995, """Exploration parameter decay rate (per episode).""")
tf.app.flags.DEFINE_integer('n_atoms', 51, """Number of n_atoms in value distribution.""")
tf.app.flags.DEFINE_float('zmin', -1.0, """Minimum of value distribution.""")
tf.app.flags.DEFINE_float('zmax', 1.0, """Maximum of value distribution.""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Examples per training update.""")
tf.app.flags.DEFINE_integer('episodes', 10000, """Episodes per train/test.""")
tf.app.flags.DEFINE_integer('update_freq', 4, """Actions/steps between updates.""")
tf.app.flags.DEFINE_integer('clone_steps', 10000, """Steps between cloning ops.""")
tf.app.flags.DEFINE_integer('max_steps', 1000, """Maximum steps per episode.""")
tf.app.flags.DEFINE_integer('min_memory_size', 10000, """Minimum number of replay memories.""")
tf.app.flags.DEFINE_integer('max_memory_size', 100000, """Maximum number of replay memories.""")
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

def log_scalar(writer, tag, value, step):
    value = [tf.Summary.Value(tag=tag, simple_value=value)]
    summary = tf.Summary(value=value)
    writer.add_summary(summary, step)

def update_p_target(p_target, indices, values):
    for i in range(p_target.shape[0]):
        for (j, v) in zip(indices[i, :], values[i, :]):
            p_target[i, j] += v
    return p_target

def train(env_name='CartPole-v0',
          device='/cpu:0',
          hidden_units=[64,64],
          learning_rate=1e-3,
          lr_decay=0.995,
          discount_factor=0.99,
          init_epsilon=1.0,
          min_epsilon=0.01,
          eps_decay=0.995,
          n_atoms=51,  # 50 segments
          zmin=-10.0,
          zmax=10.0,
          episodes=1000,
          update_freq=4,
          batch_size=32,
          clone_steps=1000,
          max_steps=1000,
          min_memory_size=10000,
          max_memory_size=100000,
          ckpt_freq=100,
          base_dir=None,  # "."
          render=True):

    # create log and checkpoint directories
    if base_dir is not None:
        ckpt_dir, log_dir = create_directories(env_name, "dqn_distributed", base_dir=base_dir)
    else:
        ckpt_dir = log_dir = None

    # create an environment
    env = gym.make(env_name)
    n_dims = state_dimensions(env)
    n_actions = available_actions(env)

    # determine value distribution grid
    values_grid, values_step = np.linspace(zmin, zmax, n_atoms, retstep=True)

    with tf.device(device):

        print('constructing graph on device: {}'.format(device))

        # create placeholders
        states_pl = tf.placeholder(tf.float32, (None, n_dims))
        actions_pl = tf.placeholder(tf.int32, (None, ))
        rewards_pl = tf.placeholder(tf.float32, (None, ))
        next_states_pl = tf.placeholder(tf.float32, (None, n_dims))
        p_target_pl = tf.placeholder(tf.float32, (None, n_atoms))
        values_pl = tf.constant(values_grid, tf.float32)  # (n_atoms,)

        # initialize networks
        logits_estimate = tf.reshape(
            mlp(states_pl, hidden_units + [n_actions * n_atoms], scope='estimate'),
            [-1, n_actions, n_atoms]
        )
        logits_target = tf.reshape(
            mlp(next_states_pl, hidden_units + [n_actions * n_atoms], scope='target'),
            [-1, n_actions, n_atoms]
        )

        P_estimate = tf.nn.softmax(logits_estimate)  # axis=-1
        P_target = tf.nn.softmax(logits_target)

        Q_estimate = tf.reduce_sum(P_estimate * values_pl, axis=-1)
        Q_target = tf.reduce_sum(P_target * values_pl, axis=-1)

        # Q_estimate = tf.reshape(
        #     tf.matmul(P_estimate, tf.reshape(values_pl, (-1, n_atoms, 1))), (-1, n_actions)
        # )
        # Q_target = tf.reshape(
        #     tf.matmul(P_target, tf.reshape(values_pl, (-1, n_atoms, 1))), (-1, n_actions)
        # )

        greedy_action = tf.arg_max(Q_estimate, dimension=1)
        target_actions = tf.arg_max(Q_target, dimension=1)

        mask_estimate = tf.one_hot(actions_pl, n_actions)
        mask_target = tf.one_hot(target_actions, n_actions)

        p_estimate = tf.reduce_sum(
            P_estimate * tf.tile(tf.reshape(mask_estimate, [-1, n_actions, 1]), [1, 1, n_atoms]),
            # P_estimate * tf.broadcast_to(mask_estimate, [n_actions, n_atoms]),
            axis=1
        )
        p_target_raw = tf.reduce_sum(  # target probabilities corresponding to target values (i.e. before projection)  [None, n_atoms]
            P_target * tf.tile(tf.reshape(mask_target, [-1, n_actions, 1]), [1, 1, n_atoms]),
            # P_target * tf.broadcast_to(mask_target, [n_actions, n_atoms]),
            axis=1
        )

        # values_target = tf.clip_by_value(rewards_pl + discount_factor * values_pl, zmin, zmax)  # clip to (zmin, zmax)
        # b = values_target / values_step
        # u = tf.cast(tf.ceil(b), tf.int32)
        # l = tf.cast(tf.floor(b), tf.int32)

        # p_target = p_target_pl + tf.reduce_sum((tf.cast(u, tf.float32) - b) * p_target_raw * tf.one_hot(l, n_atoms, axis=1), axis=-1)
        # p_target = p_target + tf.reduce_sum((b - tf.cast(l, tf.float32)) * p_target_raw * tf.one_hot(u, n_atoms, axis=1), axis=-1)

        # p_target[l] += (u - b) * p_target_raw
        # p_target[u] += (b - l) * p_target_raw
        # loss = tf.losses.cross_entropy(p_target, p_estimate)
        loss = tf.reduce_mean(-tf.reduce_sum(p_target_pl * tf.log(p_estimate), axis=-1))  # TODO: use stable cross-entropy loss!

        # merge summaries
        tf.summary.histogram('actions_pl', actions_pl)
        tf.summary.histogram('p_estimate', p_estimate)
        tf.summary.histogram('Q_estimate', Q_estimate)
        tf.summary.scalar('loss', loss)
        summary_op = tf.summary.merge_all()

        # define cloning operation
        source = tf.get_default_graph().get_collection('trainable_variables', scope='estimate')
        target = tf.get_default_graph().get_collection('trainable_variables', scope='target')
        clone_ops = [tf.assign(t, s, name='clone') for t,s in zip(target, source)]

        # initialize replay memory
        memory = create_replay_memory(env, min_memory_size, max_memory_size)

        # define training operation
        # loss = tf.losses.mean_squared_error(values, targets_pl)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

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

    def run_episode(env, sess, global_step, global_epsilon, render=False, delay=0.0):
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

            # update episode totals
            episode_reward += reward
            episode_steps += 1

            # perform update
            if episode_steps % update_freq == 0:
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = sample_memory(memory, batch_size)
                batch_p_target_raw = sess.run(p_target_raw,
                    feed_dict={
                        next_states_pl: batch_next_states,
                    })  # [batch_size, n_atoms]

                # project p_target_raw onto grid...
                batch_values = np.repeat(values_grid.reshape(1, n_atoms), batch_size, 0)
                batch_rewards = np.reshape(batch_rewards, (batch_size, 1))
                batch_dones = np.reshape(batch_dones, (batch_size, 1))
                values_target = np.clip(batch_rewards + ~batch_dones * discount_factor * batch_values, zmin, zmax)
                b = (values_target - zmin) / values_step  # b ∈ [0, n_atoms - 1]
                u = np.ceil(b).astype('int')
                l = np.floor(b).astype('int')
                batch_p_target = np.zeros([batch_size, n_atoms])
                batch_p_target = update_p_target(batch_p_target, u, (b - l) * batch_p_target_raw)
                batch_p_target = update_p_target(batch_p_target, l, (u - b) * batch_p_target_raw)

                # update
                summary, _, _ = sess.run([summary_op, loss, train_op],
                    feed_dict={
                        states_pl: batch_states,
                        actions_pl: batch_actions,
                        p_target_pl: batch_p_target
                    })

                # write log
                writer.add_summary(summary, global_step)

            # update global step count
            global_step += 1

            # update target network
            if global_step % clone_steps == 0:
                sess.run(clone_ops)
                # print("updated target network")

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
        writer = tf.summary.FileWriter(log_dir, sess.graph)  # create writer
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
            reward_history += [reward]
            elapsed_time = time.time() - t0
            avg_reward = np.mean(reward_history[-100:])
            print('episode: {:d},  reward: {:.2f},  avg. reward: {:.2f},  steps:  {:d},  epsilon: {:.2f}, lr: {:.2e},  elapsed: {:.2f}'.format(episode + 1, reward, avg_reward, global_step, global_epsilon, learning_rate, elapsed_time))

            # logging
            log_scalar(writer, 'reward', reward, global_step)
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
        E.g. ./checkpoints/CartPole-v0/dqn_distributed/GLOBAL_STEP
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
              n_atoms=FLAGS.n_atoms,
              zmin=FLAGS.zmin,
              zmax=FLAGS.zmax,
              episodes=FLAGS.episodes,
              update_freq=FLAGS.update_freq,
              batch_size=FLAGS.batch_size,
              clone_steps=FLAGS.clone_steps,
              max_steps=FLAGS.max_steps,
              ckpt_freq=FLAGS.ckpt_freq,
              min_memory_size=FLAGS.min_memory_size,
              max_memory_size=FLAGS.max_memory_size,
              base_dir=FLAGS.base_dir,
              render=FLAGS.render)
    else:
        test(env_name=FLAGS.env_name,
             hidden_units=[int(i) for i in FLAGS.hidden_units.split(',')],
             epsilon=FLAGS.min_epsilon,
             episodes=FLAGS.episodes,
             restore_dir=FLAGS.base_dir,
             render=FLAGS.render)
