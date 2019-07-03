from __future__ import print_function, division
import numpy as np
import tensorflow as tf

def create_batches(character_list, batch_size, backprop_length):
    """
        Create list of arrays with shape `[batch_size, backprop_length]`.

        # Argument
        -  `character_list`: [batch_size, ...] array
    """
    batch_size, total_length = character_list.shape
    sections, remainder = divmod(total_length, backprop_length)
    return np.split(character_list[:, :-remainder], sections, axis=-1)

def split_data(inputs, dim):
    """
        Split a batch of inputs into a list for backpropogation over time.

        # Arguments
        - `inputs`: [batch_size, backprop_length] placeholder

        # Returns
        - `inputs_series`: list of `backprop_length` [batch_size, dim] tensors
    """
    inputs_one_hots = tf.one_hot(inputs, dim)
    inputs_series = tf.split(inputs_one_hots, backprop_length, axis=1)
    inputs_series = [tf.squeeze(ip, axis=1) for ip in inputs_series]
    return inputs_series

# Configuration
num_epochs = 100
total_series_length = 50000
backprop_length = 15  # truncated backpropogation length (i.e. training sequence length)
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
batches_per_epoch = total_series_length // batch_size // backprop_length


def generate_data():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))
    return (x, y)


# Placeholders
inputs_placeholder = tf.placeholder(tf.float32, [batch_size, backprop_length])
labels_placeholder = tf.placeholder(tf.int32, [batch_size, backprop_length])
init_state = tf.placeholder(tf.float32, [batch_size, state_size])
inputs_series = tf.unstack(inputs_placeholder, axis=1)
labels_series = tf.unstack(labels_placeholder, axis=1)

# Graph
Wxs = tf.Variable(np.random.rand(1, state_size), dtype=tf.float32)
Wss = tf.Variable(np.random.rand(state_size, state_size), dtype=tf.float32)
bs = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)
Wsy = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
by = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

# Inference
state = init_state
logits_series = []
predictions_series = []
for inputs in inputs_series:
    inputs = tf.reshape(inputs, [batch_size, 1])
    state = tf.tanh(tf.matmul(inputs, Wxs) + tf.matmul(state , Wss) + bs)
    logits_series += [tf.matmul(state, Wsy) + by]
    predictions_series += [tf.nn.softmax(tf.matmul(state, Wsy) + by)]

# Backpropogation
loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_series,
        labels=labels_series
    )  # mapped across series -> [backprop_length, batch_size] tensor
)
train_op = tf.train.AdagradOptimizer(0.3).minimize(loss)

# Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch_idx in range(num_epochs):

        print("New data, epoch", epoch_idx)

        # Generate new data
        x, y = generate_data()

        # Reset initial state (optional)
        _state = np.zeros((batch_size, state_size))

        # Loop through batches
        for batch_idx in range(batches_per_epoch):

            # Find next batch
            start_idx = batch_idx * backprop_length
            end_idx = start_idx + backprop_length
            batch_inputs = x[:, start_idx:end_idx]
            batch_labels = y[:, start_idx:end_idx]

            # Perform update
            _loss, _, _state, _predictions_series = sess.run(
                [loss, train_op, state, predictions_series],
                feed_dict={
                    inputs_placeholder: batch_inputs,
                    labels_placeholder: batch_labels,
                    init_state:_state
                })

            # Report status
            if batch_idx % 100 == 0:
                print("batch: ", batch_idx, "xentropy", _loss)
