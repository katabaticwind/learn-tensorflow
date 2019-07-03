from __future__ import print_function, division
import numpy as np
import tensorflow as tf

def import_data(dir, batch_size, backprop_length):

    print('Loading text...')
    text = open(dir + 'iliad.txt').read()
    text += open(dir +'odyssey.txt').read()

    print('Creating training data...')
    text = text.replace('\n\n', ' ')  # paragraph ends and section headings
    text = text.replace('\n', ' ')  # line ends

    # split text into characters
    chars = list(text)

    # create character mappings
    tokens = np.unique(chars)
    idx_to_char = {idx:char for (idx, char) in enumerate(tokens)}
    char_to_idx = {char:idx for (idx, char) in enumerate(tokens)}
    values = [char_to_idx[c] for c in chars]

    # split characters into inputs and labels
    inputs = np.array(values[:-1])
    labels = np.array(values[1:])
    sequence_length, remainder = divmod(len(inputs), batch_size)
    inputs = inputs[:-remainder].reshape(batch_size, -1)
    labels = labels[:-remainder].reshape(batch_size, -1)

    # create batches
    inputs_batches = create_batches(inputs, batch_size, backprop_length)
    labels_batches = create_batches(labels, batch_size, backprop_length)

    # return data
    data = {'inputs': inputs_batches, 'labels': labels_batches}
    mappings = {'idx_to_char': idx_to_char, 'char_to_idx': char_to_idx}
    return data, mappings, tokens

def create_batches(character_list, batch_size, backprop_length):
    """
        Create list of arrays with shape `[batch_size, backprop_length]`.

        # Argument
        -  `character_list`: [batch_size, ...] array
    """
    batch_size, total_length = character_list.shape
    sections, remainder = divmod(total_length, backprop_length)
    return np.split(character_list[:, :-remainder], sections, axis=-1)

def split_inputs(inputs, dim):
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

def split_labels(labels):
    """
        Split a batch of labels into a list for backpropogation over time.

        # Arguments
        - `labels`: [batch_size, backprop_length] placeholder

        # Returns
        - `labels_series`: list of `backprop_length` [batch_size] tensors
    """
    labels_series = tf.split(labels, backprop_length, axis=1)
    labels_series = [tf.squeeze(ip, axis=1) for ip in labels_series]
    return labels_series

def multinomial(logits, state, temperature):
    outputs = tf.reshape(tf.pack(outputs, axis=1), [test_size, num_neurons])
    logits = tf.matmul(outputs / temperature, weights) + bias
    return logits, tf.multinomial(logits, 1)

# Configuration
dir = '../data/'
num_epochs = 100
batch_size = 32
backprop_length = 64  # truncated backpropogation length (i.e. training sequence length)
num_classes = 67  # len(tokens)
state_size = 256
temp = 1.5
lr = 1e-3

# Import data
data, mappings, tokens = import_data(dir, batch_size, backprop_length)
inputs_batches = data['inputs']
labels_batches = data['labels']
idx_to_char = mappings['idx_to_char']
batches_per_epoch = len(inputs_batches)

# Placeholders
inputs_placeholder = tf.placeholder(tf.int32, [batch_size, backprop_length])
labels_placeholder = tf.placeholder(tf.int32, [batch_size, backprop_length])
init_state = tf.placeholder(tf.float32, [batch_size, state_size])
inputs_series = split_inputs(inputs_placeholder, num_classes)
labels_series = split_labels(labels_placeholder)

# Graph
Wxs = tf.Variable(np.random.rand(num_classes, state_size), dtype=tf.float32)
Wss = tf.Variable(np.random.rand(state_size, state_size), dtype=tf.float32)
bs = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)
Wsy = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
by = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

# Inference
state = init_state
logits_series = []
predictions_series = []
for inputs in inputs_series:
    state = tf.tanh(tf.matmul(inputs, Wxs) + tf.matmul(state , Wss) + bs)
    logits_series += [tf.matmul(state, Wsy) + by]
    # predictions_series += [tf.nn.softmax(tf.matmul(state, Wsy) + by)]
    predictions_series += [tf.multinomial(tf.matmul(state / temp, Wsy) + by, 1)]
predictions = tf.squeeze(tf.stack(predictions_series, axis=1), axis=-1)

# Backpropogation
loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_series,
        labels=labels_series
    )  # mapped across series -> [backprop_length, batch_size] tensor
)
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch_idx in range(num_epochs):

        print(f"epoch={epoch_idx}")

        # Reset initial state (optional)
        _state = np.zeros((batch_size, state_size))

        # Loop through batches
        for batch_idx in range(batches_per_epoch):

            # get batch data
            batch_inputs = inputs_batches[batch_idx]
            batch_labels = labels_batches[batch_idx]

            # Perform update
            _loss, _, _state, _predictions = sess.run(
                [loss, train_op, state, predictions],
                feed_dict={
                    inputs_placeholder: batch_inputs,
                    labels_placeholder: batch_labels,
                    init_state:_state
                })

            if batch_idx == 0:
                l = batch_labels[0, :]
                p = _predictions[0, :]
                print(''.join([idx_to_char[i] for i in l]))
                print(''.join([idx_to_char[i] for i in p]))

            # Report status
            if batch_idx % 25 == 0:
                print("batch: ", batch_idx, "xentropy", _loss)
