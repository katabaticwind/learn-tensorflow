import tensorflow as tf
import numpy as np
import gym
import time

env_name = 'CartPole-v0'
device = '/cpu:0'
lr = 1e-3
batch_size = 5000
gamma = 1.00
state_dim = 4
action_dim = 2
sizes = [32]

def mlp(x, sizes, activation, output_activation=None):
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)
    if sizes[-1] == 1:
        return tf.squeeze(tf.layers.dense(x, units=sizes[-1], activation=output_activation))
    else:
        return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def train():

    with tf.device(device):
        states_pl = tf.placeholder(tf.float32, [None, state_dim])
        actions_pl = tf.placeholder(tf.int32, [None])
        rewards_pl = tf.placeholder(tf.float32, [None])
        flags_pl = tf.placeholder(tf.int32, [None])
        targets_pl = tf.placeholder(tf.float32, [None])
        values = mlp(states_pl, sizes + [1], tf.tanh)  # V(s)
        policy_logits = mlp(states_pl, sizes + [action_dim], tf.tanh)
        action_sample = tf.squeeze(tf.multinomial(logits=policy_logits, num_samples=1), axis=1)
        action_mask = tf.one_hot(actions_pl, action_dim)
        policy = tf.reduce_sum(action_mask * tf.nn.log_softmax(policy_logits), axis=1)  # π(a | s)
        value_targets = rewards_pl + gamma * values * tf.cast(flags_pl, tf.float32)  # r + γ * V(s')
        policy_targets = targets_pl - values  # A(s, a) = Q(s, a) - V(s) = r + γ * V(s') - V(s)
        value_loss = tf.losses.mean_squared_error(targets_pl, values)
        policy_loss = -tf.reduce_mean(policy * targets_pl)
        value_update = tf.train.AdamOptimizer(learning_rate=lr).minimize(value_loss)
        policy_update = tf.train.AdamOptimizer(learning_rate=lr).minimize(policy_loss)
        init_op = tf.global_variables_initializer()

    def collect_experience():
        states = []
        actions = []
        rewards = []
        next_states = []
        done_flags = []
        batch_returns = []
        while len(states) < batch_size:
            state = env.reset()
            episode_return = 0.0
            start_time = time.time()
            while True:
                action = sess.run(action_sample, feed_dict={states_pl: state.reshape(1, -1)})[0]
                next_state, reward, done, info = env.step(action)
                states += [state.copy()]
                actions += [action]
                rewards += [reward]
                next_states += [next_state.copy()]
                done_flags += [done]
                state = next_state  # update state!
                episode_return += reward
                if done:
                    batch_returns += [episode_return]
                    break
        print(f"batch_return={np.mean(batch_returns)}")
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done_flags)

    with tf.Session() as sess:
        sess.run(init_op)
        env = gym.make(env_name)
        while True:
            states, actions, rewards, next_states, done_flags = collect_experience()
            targets = sess.run(value_targets, feed_dict={states_pl: next_states, rewards_pl: rewards, flags_pl: ~done_flags})
            sess.run(value_update, feed_dict={states_pl: states, targets_pl: targets})
            targets = sess.run(policy_targets, feed_dict={states_pl: states, targets_pl: targets})
            sess.run(policy_update, feed_dict={states_pl: states, actions_pl: actions, targets_pl: targets})

if __name__ == '__main__':
    train()
