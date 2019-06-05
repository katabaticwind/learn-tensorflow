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

def crop_frame(frame):
    """Reduce size of frame."""
    return cv.resize(frame, (84, 84))

def rgb_to_luminance(frame):
    """Calculate the relative luminance of an RGB image/frame."""
    frame_rgb = (frame / 255).astype(np.float32)  # convert to [0, 1] range!
    frame_lum = 0.2126 * frame_rgb[:, :, 0] + 0.7152 * frame_rgb[:, :, 1] + 0.0722 * frame_rgb[:, :, 2]
    return frame_lum.reshape(frame_lum.shape + (1,))

def preprocess(frame):
    return rgb_to_luminance(crop_frame(frame))

def collect_frames(queue, nframes):
    queue.fill(nframes)
    return np.concatenate(queue.queue, axis=2)






def convert_frames(frames, min_len=4):
    """Convert collection of frames to states."""
    while len(frames) < min_len:
        frames.appendleft(frames[0])
    return np.concatenate([rgb_to_luminance(crop_frame(X)) for X in frames], axis=2)

def append_to_queue(queue, value, max_len=4):
    """Add a new value to `queue` with maximum length `max_len`."""
    if len(queue) == max_len:
        queue.popleft()  # queue[0] is oldest value
        queue.append(value)
    else:
        queue.append(value)

def example():
    env = gym.make('Breakout-v0')
    state = env.reset()
    queue = deque([state])
    states = []
    states.append(convert_frames(queue))
    for _ in range(3):
        state, reward, done, info = env.step(env.action_space.sample())
        append_to_queue(queue, state)
        states.append(convert_frames(queue))
    for i in range(4):
        plt.subplot(2, 4, i + 1)
        plt.imshow(states[3][:, :, i], cmap="gray")
        plt.subplot(2, 4, i + 1 + 4)
        plt.imshow(queue[i])
    plt.tight_layout()
    plt.show()
    return queue, states
