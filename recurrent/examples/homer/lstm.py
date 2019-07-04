from __future__ import print_function, division
import numpy as np
import tensorflow as tf

def gate(W, x, U, h, b, activation):
    return activation(tf.matmul(x, W) + tf.matmul(h, U) + b)

class LSTM(object):
    """Basic LSTM cell."""

    def __init__(self, input_dim, hidden_dim, use_dropout=True):

        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.Wf = tf.Variable(
            tf.truncated_normal([input_dim, hidden_dim], stddev=0.02),
            dtype=tf.float32,
            name='weights_forget_gate'
        )
        self.Wi = tf.Variable(
            tf.truncated_normal([input_dim, hidden_dim], stddev=0.02),
            dtype=tf.float32,
            name='weights_input_gate'
        )
        self.Wo = tf.Variable(
            tf.truncated_normal([input_dim, hidden_dim], stddev=0.02),
            dtype=tf.float32,
            name='weights_output'
        )
        self.Wm = tf.Variable(
            tf.truncated_normal([input_dim, hidden_dim], stddev=0.02),
            dtype=tf.float32,
            name='weights_memory'
        )

        self.Uf = tf.Variable(
            tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.02),
            dtype=tf.float32,
            name='U_forget'
        )
        self.Ui = tf.Variable(
            tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.02),
            dtype=tf.float32,
            name='U_input'
        )
        self.Uo = tf.Variable(
            tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.02),
            dtype=tf.float32,
            name='U_output'
        )
        self.Um = tf.Variable(
            tf.truncated_normal([hidden_dim, hidden_dim], stddev=0.02),
            dtype=tf.float32,
            name='U_memory'
        )

        self.bf = tf.Variable(
            tf.constant(0.0, shape=[hidden_dim]),
            name='bias_forget'
        )
        self.bi = tf.Variable(
            tf.constant(0.0, shape=[hidden_dim]),
            name='bias_input'
        )
        self.bo = tf.Variable(
            tf.constant(0.0, shape=[hidden_dim]),
            name='bias_output'
        )
        self.bm = tf.Variable(
            tf.constant(0.0, shape=[hidden_dim]),
            name='bias_memory'
        )

        self.use_dropout = use_dropout

    def forward(self, inputs, init_hidden, init_memory, keep_prob):
        """Construct graph for training LSTM.

            # Arguments
            - `inputs`: list of [None, input_dim] placeholders
            - `init_hidden`: [None, hidden_dim] placeholder representing initial hidden state
            - `init_memory`: [None, hidden_dim] placeholder representing initial memory state
            - `keep_prob (tf.placeholder)`: placeholder representing the dropout keep probability

            # Returns
            - `logits`: list of [None, output_dim] tensors
            - `hidden`: [None, hidden_dim] tensor representing final hidden state
            - `memory`: [None, hidden_dim] tensor representing final memory state
        """
        hidden = init_hidden
        memory = init_memory
        outputs = []
        for input in inputs:
            f = gate(self.Wf, input, self.Uf, hidden, self.bf, tf.sigmoid)
            i = gate(self.Wi, input, self.Ui, hidden, self.bi, tf.sigmoid)
            o = gate(self.Wo, input, self.Uo, hidden, self.bo, tf.sigmoid)
            m = gate(self.Wm, input, self.Um, hidden, self.bm, tf.tanh)
            memory = tf.multiply(f, memory) + tf.multiply(i, m)
            hidden = tf.multiply(o, tf.tanh(memory))
            if self.use_dropout:
                output += [tf.nn.dropout(hidden, keep_prob)]
            else:
                outputs += [hidden]
        return outputs, hidden, memory

    def predict(self, input, hidden, memory):
        """Construct graph for predicting from an input.

            # Arguments
            - `input`: [None, input_dim] placeholder
            - `hidden`: [None, hidden_dim] placeholder representing hidden state
            - `memory`: [None, hidden_dim] placeholder representing memory state

            # Returns
            - `prediction`: [None, output_dim] tensor representing predicted output
            - `next_hidden`: [None, hidden_dim] tensor representing next hidden state
            - `next_memory`: [None, hidden_dim] tensor representing next memory state
        """
        f = gate(self.Wf, input, self.Uf, hidden, self.bf, tf.sigmoid)
        i = gate(self.Wi, input, self.Ui, hidden, self.bi, tf.sigmoid)
        o = gate(self.Wo, input, self.Uo, hidden, self.bo, tf.sigmoid)
        m = gate(self.Wm, input, self.Um, hidden, self.bm, tf.tanh)
        next_memory = tf.multiply(f, memory) + tf.multiply(i, m)
        next_hidden = tf.multiply(o, tf.tanh(memory))
        return next_hidden, next_memory

class Chain(object):

    def __init__(self, cells):

        super(Chain, self).__init__()

        self.cells = cells

    def forward(self, inputs, init_hiddens, init_memories, keep_prob):
        """
            # Arguments
            - `inputs (list<tf.placeholder>)`: `backprop_length` list of `[batch_size, input_dim]` placeholders
            - `init_hidden (list<tf.placeholder>)`: `len(cells)` list of `[batch_size, hidden_dim]` placeholders
            - `init_memory (list<tf.placeholder>)`: `len(cells)` list of `[batch_size, hidden_dim]` placeholders

            # Returns
            - `outputs (list<tf.tensor>)`: `backprop_length` list of `[batch_size, input_dim]` tensors
            - `hidden (list<tf.tensor>)`: `len(cells)` list of `[batch_size, hidden_dim]` tensors
            - `memory (list<tf.tensor>)`: `len(cells)` list of `[batch_size, hidden_dim]` tensors
        """

        hidden_list = []
        memory_list = []
        for (cell, init_hidden, init_memory) in zip(self.cells, init_hiddens, init_memories):
            outputs, hidden, memory = cell.forward(inputs, init_hidden, init_memory, keep_prob)
            inputs = outputs
            hidden_list += [hidden]
            memory_list += [memory]
        return outputs, hidden_list, memory_list

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

def backward(logits, labels, learning_rate):
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels
        )
    )
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)
    return loss, train_op

# Configuration
dir = '../../data/'
num_epochs = 100
batch_size = 32
backprop_length = 64  # truncated backpropogation length (i.e. training sequence length)
input_dim = 67  # len(tokens)
output_dim = 67  # len(tokens)
hidden_dim = 64
temp = 1.0
learning_rate = 1e-3

# Import data
data, mappings, tokens = import_data(dir, batch_size, backprop_length)
inputs_batches = data['inputs']
labels_batches = data['labels']
idx_to_char = mappings['idx_to_char']
batches_per_epoch = len(inputs_batches)

# Placeholders
inputs_placeholder = tf.placeholder(tf.int32, [batch_size, backprop_length])
labels_placeholder = tf.placeholder(tf.int32, [batch_size, backprop_length])
init_hidden = [
    tf.placeholder(tf.float32, [batch_size, hidden_dim]),
    tf.placeholder(tf.float32, [batch_size, hidden_dim])
]
init_memory = [
    tf.placeholder(tf.float32, [batch_size, hidden_dim]),
    tf.placeholder(tf.float32, [batch_size, hidden_dim])
]
inputs_series = split_inputs(inputs_placeholder, output_dim)
labels_series = split_labels(labels_placeholder)
keep_pl = tf.placeholder(tf.float32)

# Graph
print("Constructing graph...")
lstm_1 = LSTM(input_dim=input_dim, hidden_dim=hidden_dim)
lstm_2 = LSTM(input_dim=hidden_dim, hidden_dim=hidden_dim)
lstm = Chain([lstm_1, lstm_2])
outputs_series, hidden, memory = lstm.forward(
    inputs_series,
    init_hidden,
    init_memory,
    keep_pl
)
Why = tf.Variable(
    tf.truncated_normal([hidden_dim, output_dim], stddev=0.02),
    dtype=tf.float32,
    name='weights_softmax'
)
by = tf.Variable(
    tf.constant(0.0, shape=[output_dim]),
    name='bias_softmax'
)
logits_series = [tf.matmul(h, Why) + by for h in outputs_series]
loss, train_op = backward(logits_series, labels_series, learning_rate)

# Loop
print("Beginning training...")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch_idx in range(num_epochs):

        print(f"**epoch={epoch_idx}**")

        # Reset initial state (optional)
        _hidden = [
            np.zeros((batch_size, hidden_dim)),
            np.zeros((batch_size, hidden_dim))
        ]
        _memory = [
            np.zeros((batch_size, hidden_dim)),
            np.zeros((batch_size, hidden_dim))
        ]

        # Loop through batches
        for batch_idx in range(batches_per_epoch):

            # get batch data
            batch_inputs = inputs_batches[batch_idx]
            batch_labels = labels_batches[batch_idx]

            # Perform update
            feed_dict = {}
            feed_dict[init_hidden[0]] = _hidden[0]
            feed_dict[init_hidden[1]] = _hidden[1]
            feed_dict[init_memory[0]] = _memory[0]
            feed_dict[init_memory[1]] = _memory[1]
            feed_dict[inputs_placeholder] = batch_inputs
            feed_dict[labels_placeholder] = batch_labels
            _loss, _, _hidden, _memory = sess.run(
                [loss, train_op, hidden, memory],
                feed_dict=feed_dict
            )

            # Report status
            if batch_idx % 10 == 0:
                print("batch: ", batch_idx, "xentropy", _loss)
