import tensorflow as tf
import numpy as np
import gym
import time
from datetime import datetime

# TODO: try using N-step TD-errors
# TODO: try using cloned value network

env_name = 'CartPole-v0'
device = '/cpu:0'
log_freq = 10
log_dir = './logs/a2c/'
lr = 1e-2
beta = 0.01
gamma = 1.00
state_dim = 4
action_dim = 2
sizes = [32]
max_episodes = 5000
max_steps = 200
pass_condition = 195.0

def calculate_targets(continuation_value, rewards):
    """Calculate targets used to update policy and value functions."""
    targets = []
    R = continuation_value
    for r in rewards[-2::-1]:
        R = r + gamma * R
        targets += [R]
    return targets[-1::-1]  # reverse to match original ordering


def mlp(x, sizes, activation, output_activation=None):
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)
    if sizes[-1] == 1:
        return tf.squeeze(tf.layers.dense(x, units=sizes[-1], activation=output_activation))
    else:
        return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def log_scalar(writer, tag, value, step):
    value = [tf.Summary.Value(tag=tag, simple_value=value)]
    summary = tf.Summary(value=value)
    writer.add_summary(summary, step)

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
        value_loss = tf.reduce_mean(tf.square(targets_pl - values))
        policy_loss = -tf.reduce_mean(policy * targets_pl)
        entropy_loss = beta * tf.reduce_mean(
            tf.multiply(
                tf.nn.softmax(policy_logits),  # probabilities
                tf.nn.log_softmax(policy_logits)  # log probabilities
            )
        )
        value_update = tf.train.AdamOptimizer(learning_rate=lr).minimize(value_loss)
        policy_update = tf.train.AdamOptimizer(learning_rate=lr).minimize(policy_loss + entropy_loss)
        init_op = tf.global_variables_initializer()

    # tensorboard
    tf.summary.histogram('policy_weights', policy_targets)
    tf.summary.histogram('policy_logits', policy_logits)
    summary_op = tf.summary.merge_all()

    def update_networks(states, actions, rewards, next_states, done_flags):

            # calculate value targets
            v_targets = sess.run(
                value_targets,
                feed_dict={
                    states_pl: np.array(next_states),
                    rewards_pl: np.array(rewards),
                    flags_pl: ~np.array(done_flags)
                }
            )

            # update value function
            sess.run(
                value_update,
                feed_dict={
                    states_pl: np.array(states),
                    targets_pl: v_targets
                }
            )

            # calculate policy weights
            v_targets = sess.run(
                value_targets,
                feed_dict={
                    states_pl: np.array(next_states),
                    rewards_pl: np.array(rewards),
                    flags_pl: ~np.array(done_flags)
                }
            )
            p_targets = sess.run(
                policy_targets,
                feed_dict={
                    states_pl: np.array(states),
                    targets_pl: v_targets
                }
            )

            # upate policy function
            summary, _ = sess.run(
                [summary_op, policy_update],
                feed_dict={
                    states_pl: np.array(states),
                    actions_pl: np.array(actions),
                    targets_pl: p_targets
                }
            )
            return summary

    def run_episode():
        states = []
        actions = []
        rewards = []
        next_states = []
        done_flags = []
        state = env.reset()
        start_time = time.time()
        episode_return = 0.0
        episode_steps = 0
        while True:
            action = sess.run(action_sample, feed_dict={states_pl: state.reshape(1, -1)})[0]
            next_state, reward, done, info = env.step(action)
            states += [state]
            actions += [action]
            rewards += [reward]
            next_states += [next_state]
            done_flags += [done]
            state = next_state
            episode_return += reward
            episode_steps += 1
            if done:
                summary = update_networks(states, actions, rewards, next_states, done_flags)
                elapsed_time = time.time() - start_time
                return episode_return, elapsed_time, episode_steps, summary

    with tf.Session() as sess:
        sess.run(init_op)
        now = datetime.today()
        date_string = now.strftime("%Y-%m-%d-%H:%M:%S.%f")
        writer = tf.summary.FileWriter(log_dir + '/' + date_string, sess.graph)
        env = gym.make(env_name)
        returns = []
        avg_returns = []
        steps = [0]
        for episode in range(max_episodes):
            episode_return, episode_time, episode_steps, summary = run_episode()
            returns += [episode_return]
            avg_returns += [np.mean(np.array(returns)[-100:])]
            steps += [steps[-1] + episode_steps]
            if episode % log_freq == 0 or avg_returns[-1] > pass_condition:
                print(f"episode={episode}, avg_return={avg_returns[-1]:.2f}, total_steps={steps[-1]}")
                log_scalar(writer, 'reward', episode_return, steps[-1])
                log_scalar(writer, 'steps', episode_steps, steps[-1])
                log_scalar(writer, 'steps_per_second', episode_steps / episode_time, steps[-1])
                log_scalar(writer, 'avg_reward', avg_returns[-1], steps[-1])
                writer.add_summary(summary, steps[-1])
                if avg_returns[-1] > pass_condition:
                    print("reached pass condition!")
                    break

if __name__ == '__main__':
    train()
