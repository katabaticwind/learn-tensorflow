import gym, time

"""

"""

def test_environment(env_name, render=True):
    """Run a sample episode of environment using a random action policy."""
    env = gym.make(env_name)
    state = env.reset()
    total_reward = 0
    if render:
        env.render()
        time.sleep(0.01)
    while True:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward
        if render:
            env.render()
            time.sleep(0.01)
        if done:
            break
    return total_reward

def available_actions(env):
    # if type(env.action_space) == gym.spaces.discrete.Discrete:
    try:
        return env.action_space.n
    except AttributeError:
        raise AttributeError("env.action_space is not Discrete")

def state_dimensions(env):
    """Find the number of dimensions in the state."""
    return env.observation_space.shape[0]

if __name__ == '__main__':
    test_environment('Breakout-v0')
