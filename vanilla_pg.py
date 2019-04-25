import tensorflow as tf
import numpy as np
import gym
from gym.spaces import Discrete, Box

# TODO: saving checkoints
# TODO: re-write mlp with low-level API (tf.Variables)

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    """Build a feedforward neural network."""
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)
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

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    logits = mlp(obs_ph, sizes=hidden_sizes+[n_acts])

    # make action selection op (outputs int actions, sampled from policy)
    actions = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)

    # make loss function whose gradient, for the right data, is policy gradient
    weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    action_masks = tf.one_hot(act_ph, n_acts)
    log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
    loss = -tf.reduce_mean(weights_ph * log_probs)

    # make train op
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())  # copy original state

            # act in the environment
            act = sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]  # need obs to be 1 x nstates
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                # batch_weights += [ep_ret] * ep_len  # using full reward for each step
                batch_weights += reward_to_go(ep_rews)

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        batch_loss, _ = sess.run([loss, train_op],
                                 feed_dict={
                                    obs_ph: np.array(batch_obs),
                                    act_ph: np.array(batch_acts),
                                    weights_ph: np.array(batch_weights)
                                 })
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

def train_2(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, batches=50, batch_size=5000, render=False):

    # create placeholders
    rewards_pl = tf.placeholder(tf.float32, (None, ))
    states_pl = tf.placeholder(tf.float32, (None, 4))
    actions_pl = tf.placeholder(tf.float32, (None, ))

    # create a policy network
    logits = mlp(states_pl, hidden_sizes + [2])
    actions = tf.squeeze(tf.random.categorical(logits=logits, num_samples=1, dtype=tf.int32), axis=1)  # chooses an action

    # define training operation
    actions_mask = tf.one_hot(tf.cast(actions_pl, tf.int32), 2)
    log_probs = tf.reduce_sum(actions_mask * tf.nn.log_softmax(logits), axis=1)  # use tf.mask instead?
    loss = -tf.reduce_mean(rewards_pl * log_probs)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # create an environment
    env = gym.make('CartPole-v0')

    def run_episode(env, sess):
        state = env.reset()
        episode_states = [state]
        episode_actions = []
        total_reward = 0
        # env.render()
        while True:
            action = sess.run(actions, feed_dict={states_pl: state.reshape(1, -1)})  # state is (4,), state_pl requires (None, 4)
            state, reward, done, info = env.step(action[0])  # step requires scalar action
            # env.render()
            episode_states += [state.copy()]  # no reshape b/c we will convert to np.array later...
            episode_actions += [action[0]]
            total_reward += reward
            if done:
                break
        return episode_states, episode_actions, [total_reward] * len(episode_actions)

    def run_batch(env, sess):
        batch_states = []
        batch_actions = []
        batch_rewards = []
        while len(batch_rewards) < batch_size:
            episode_states, episode_actions, episode_rewards = run_episode(env, sess)
            batch_states.extend(episode_states[:-1])  # only keep states preceeding each action
            batch_actions.extend(episode_actions)
            batch_rewards.extend(episode_rewards)
        return batch_states, batch_actions, batch_rewards

    def update_network(env, states, actions, rewards, sess):
        feed_dict = {
            states_pl: states,
            rewards_pl: rewards,
            actions_pl: actions
        }
        batch_loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)
        return batch_loss

    # train the network
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for batch in range(batches):

            batch_states, batch_actions, batch_rewards = run_batch(env, sess)

            batch_loss = update_network(env, batch_states, batch_actions, batch_rewards, sess)

            print("batch = {:d}, mean_reward = {:.2f}".format(batch + 1, np.mean(batch_rewards)))

if __name__ == '__main__':
    train(env_name='CartPole-v0', render=False, lr=1e-2)
