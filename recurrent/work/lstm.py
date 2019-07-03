import tensorflow as tf
import numpy as np
from datetime import datetime

def train():
    """Train LSTM network on character-level data."""

    # prepare data

    # construct graph

    # begin training
    for epoch in range(max_epochs):
        for batch in range(batches_per_epoch):

            # perform update


            # log status


            # save checkpoint


def test():
    """Test LSTM network on character-level data."""
    pass


# Placeholder for the inputs in a given iteration.
words = tf.placeholder(tf.float32, [batch_size, input_dim, num_steps])

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
# Initial state of the LSTM memory.
initial_state = state = lstm.zero_state(batch_size, dtype=tf.float32)

for i in range(num_steps):
    # The value of state is updated after processing each batch of words.
    output, state = lstm(words[:, :, i], state)

    # The rest of the code.
    # ...

final_state = state
