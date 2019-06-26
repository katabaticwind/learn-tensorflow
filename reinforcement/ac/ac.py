import tensorflow as tf
import numpy as np
import gym
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

env_name = 'CartPole-v0'
device = '/cpu:0'
lr = 1e-2
batch_size = 32
gamma = 1.00
state_dim = 4
action_dim = 2
sizes = [32]
max_steps = 5000

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
        batch_episodes = 0
        while len(states) < batch_size:
            state = env.reset()
            episode_return = 0.0
            start_time = time.time()
            while True:
                action = sess.run(action_sample, feed_dict={states_pl: state.reshape(1, -1)})[0]
                next_state, reward, done, info = env.step(action)
                states += [state]
                actions += [action]
                rewards += [reward]
                next_states += [next_state]
                done_flags += [done]
                state = next_state  # update state!
                episode_return += reward
                if done:
                    batch_returns += [episode_return]
                    batch_episodes += 1
                    break
        # print(f"batch_return={np.mean(batch_returns)}, batch_episodes={batch_episodes}, batch_steps={len(states)}")
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done_flags), np.mean(batch_returns)

    with tf.Session() as sess:
        sess.run(init_op)
        env = gym.make(env_name)
        returns = []
        ma_returns = []
        steps = 0
        xvals = []
        for batch in range(max_steps):
            states, actions, rewards, next_states, done_flags, mean_returns = collect_experience()
            returns += [mean_returns]
            ma_returns += [np.mean(np.array(returns)[-100:])]
            steps += len(states)
            xvals += [steps]

            # calculate value targets
            v_targets = sess.run(
                value_targets,
                feed_dict={
                    states_pl: next_states,
                    rewards_pl: rewards,
                    flags_pl: ~done_flags
                }
            )

            # update value function
            sess.run(
                value_update,
                feed_dict={
                    states_pl: states,
                    targets_pl: v_targets
                }
            )

            v_targets = sess.run(
                value_targets,
                feed_dict={
                    states_pl: next_states,
                    rewards_pl: rewards,
                    flags_pl: ~done_flags
                }
            )
            p_targets = sess.run(
                policy_targets,
                feed_dict={
                    states_pl: states,
                    targets_pl: v_targets
                }
            )

            # calculate policy targets
            # targets = sess.run(
            #     value_targets,
            #     feed_dict={
            #         states_pl: next_states,
            #         rewards_pl: rewards,
            #         flags_pl: ~done_flags
            #     }
            # )
            # targets = sess.run(
            #     policy_targets,
            #     feed_dict={
            #         states_pl: states,
            #         targets_pl: targets
            #     }
            # )

            # upate policy function
            sess.run(
                policy_update,
                feed_dict={
                    states_pl: states,
                    actions_pl: actions,
                    targets_pl: p_targets
                }
            )

            if batch % 25 == 0:
                print(f"batch={batch}, steps={steps}, value_target={np.mean(v_targets):.2f}, policy_target={np.mean(p_targets):.2f}, avg_return={ma_returns[-1]:.2f}")
                plt.plot(xvals, np.array(returns), color='C0')
                plt.plot(xvals, np.array(ma_returns), color='C1')
                # plt.xlim(0, max_steps)
                plt.ylim(0, 250)
                plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    train()
