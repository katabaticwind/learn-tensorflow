import numpy as np
import gym
from collections import deque
import cv2 as cv

"""
Codes to pre-process Atari environment frames.

Basic formula to convert observations (frames) to "states":
    - Crop image.
    - Transform RGB to "luminance".
    - Collect M consecutive frames (M = 2, 3, or 4).

A possible strategy is to maintain a queue (i.e., `collections.deque`) of maximum length `M` frames during each episode. After each step, append the new frame and drop the first frame in the queue, then apply the basic formula to the queue. If there are only `m < M` frames, the last step of the formula repeats the oldest of these frames up to `M - 1` times (for a total of `M` frames). The advantage of this strategy is that you have to carry out preprocessing during training no matter what (to give proper input to policy or action-value network), so we avoid redundancy of preprocesing frames a second time during parameter updates.
"""

def crop_frame(X):
    """Reduce size of frame.

        Making height 208 pixels allows us to evenly down-sample multiple times during forward pass. E.g., if each convolutional layer uses max-pooling such that the height and width are halved, then the feature maps have sizes (104, 80), (52, 40), (26, 20), and (13, 10) following four convolutional layers.
    """
    # return X[1:-1, :, :]  # 208 x 160 x 3
    return cv.resize(X, (84, 84))

def RGB_to_luminance(X):
    """Calculate the relative luminance of an RGB image/frame."""
    X = (X / 255).astype(np.float32)  # convert to [0, 1] range!
    L = 0.2126 * X[:, :, 0] + 0.7152 * X[:, :, 1] + 0.0722 * X[:, :, 2]
    return L.reshape(L.shape + (1,))

def frames_to_state(frame_queue, min_len=4):
    """Convert collection of frames to states."""
    frame_queue.fill(min_len)
    return np.concatenate([RGB_to_luminance(crop_frame(X)) for X in frame_queue.queue], axis=2)


import matplotlib
matplotlib.use('TkAgg')  # bug w/ MacOS
import matplotlib.pyplot as plt

"""
Check that the code is working using a simple example.
"""
def example():

    def append_to_queue(queue, value, max_len=4):
        """Add a new value to `queue` with maximum length `max_len`."""
        if len(queue) == max_len:
            queue.popleft()  # queue[0] is oldest value
            queue.append(value)
        else:
            queue.append(value)

    env = gym.make('Breakout-v0')
    state = env.reset()
    queue = deque([state])
    states = []
    states.append(frames_to_state(queue))

    # get 3 more frames...
    for _ in range(3):
        state, reward, done, info = env.step(env.action_space.sample())
        append_to_queue(queue, state)
        states.append(frames_to_state(queue))

    # plot the 4 frames, plus the most recent frame in each state
    for i in range(4):
        plt.subplot(2, 4, i + 1)
        plt.imshow(states[3][:, :, i], cmap="gray")
        plt.subplot(2, 4, i + 1 + 4)
        plt.imshow(queue[i])
    plt.tight_layout()
    plt.show()
    return queue, states

import time
def Pong(delay=0.0):
    env = gym.make('Pong-v0')
    env.reset()
    env.render()
    time.sleep(delay)
    actions = []
    rewards = []
    while True:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        actions += [action]
        rewards += [reward]
        env.render()
        time.sleep(delay)
        if done:
            break
    env.close()
    return np.array(actions), np.array(rewards)
