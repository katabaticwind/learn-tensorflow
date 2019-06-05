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

# def crop_frame(frame):
#     """Reduce size of frame."""
#     return cv.resize(frame, (84, 84))
#
# def rgb_to_luminance(frame):
#     """Calculate the relative luminance of an RGB image/frame."""
#     frame_rgb = (frame / 255).astype(np.float32)  # convert to [0, 1] range!
#     frame_lum = 0.2126 * frame_rgb[:, :, 0] + 0.7152 * frame_rgb[:, :, 1] + 0.0722 * frame_rgb[:, :, 2]
#     return frame_lum.reshape(frame_lum.shape + (1,))
#
# def preprocess(frame):
#     return rgb_to_luminance(crop_frame(frame))

def collect_frames(queue, nframes):
    queue.fill(nframes)
    return np.stack(queue.queue, axis=-1)

import tensorflow as tf
def rgb_to_grayscale(frame):
    frame_grayscale = tf.image.rgb_to_grayscale(frame)
    frame_cropped = tf.image.crop_to_bounding_box(frame_grayscale, 34, 0, 160, 160)
    frame_resized = tf.image.resize_images(frame_cropped, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf.squeeze(frame_resized)

# def preprocess(frame, sess):
#     frame_processed = sess.run(preprocess_op, {frame_pl: frame})
