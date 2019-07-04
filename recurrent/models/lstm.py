from __future__ import print_function, division
import tensorflow as tf
import numpy as np

def gate(W, x, U, h, b, activation):
    return activation(tf.matmul(x, W) + tf.matmul(h, U) + b)

class LSTM(object):
    """Basic LSTM cell."""

    def __init__(self, input_dim, hidden_dim):

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

    def forward(self, inputs, init_hidden, init_memory):
        """Construct graph for training LSTM.

            # Arguments
            - `inputs`: list of [None, input_dim] placeholders
            - `init_hidden`: [None, hidden_dim] placeholder representing initial hidden state
            - `init_memory`: [None, hidden_dim] placeholder representing initial memory state

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

    def forward(self, inputs, init_hiddens, init_memories):
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
            outputs, hidden, memory = cell.forward(inputs, init_hidden, init_memory)
            inputs = outputs
            hidden_list += [hidden]
            memory_list += [memory]
        return outputs, hidden_list, memory_list


def test():

    num_epochs = 20
    total_series_length = 50000
    backprop_length = 15  # truncated backpropogation length (i.e. training sequence length)
    input_size = 1
    state_size = 4
    num_classes = 2
    echo_step = 3
    batch_size = 5
    learning_rate = 1e-2
    batches_per_epoch = total_series_length // batch_size // backprop_length

    def generate_batches():
        x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
        y = np.roll(x, echo_step)
        y[0:echo_step] = 0
        x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
        y = y.reshape((batch_size, -1))
        return (x, y)

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

    # Placeholders
    inputs_placeholder = tf.placeholder(tf.float32, [batch_size, backprop_length])
    labels_placeholder = tf.placeholder(tf.int32, [batch_size, backprop_length])
    init_hidden = tf.placeholder(tf.float32, [batch_size, state_size])
    init_memory = tf.placeholder(tf.float32, [batch_size, state_size])
    inputs_series = tf.unstack(inputs_placeholder, axis=1)
    inputs_series = [tf.reshape(i, [batch_size, input_size]) for i in inputs_series]
    labels_series = tf.unstack(labels_placeholder, axis=1)

    # Graph
    lstm = LSTM(input_size, state_size)
    outputs_series, hidden, memory = lstm.forward(inputs_series, init_hidden, init_memory)
    Why = tf.Variable(
        tf.truncated_normal([state_size, num_classes], stddev=0.02),
        dtype=tf.float32,
        name='weights_forget_gate'
    )
    by = tf.Variable(
        tf.constant(0.0, shape=[num_classes]),
        name='bias_forget'
    )
    logits_series = [tf.matmul(h, Why) + by for h in outputs_series]
    loss, train_op = backward(logits_series, labels_series, learning_rate)

    # Loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_idx in range(num_epochs):

            print("New data, epoch", epoch_idx)

            # Generate new data
            x, y = generate_batches()

            # Reset initial state (optional)
            _hidden = np.zeros((batch_size, state_size))
            _memory = np.zeros((batch_size, state_size))

            # Loop through batches
            for batch_idx in range(batches_per_epoch):

                # Find next batch
                start_idx = batch_idx * backprop_length
                end_idx = start_idx + backprop_length
                batch_inputs = x[:, start_idx:end_idx]
                batch_labels = y[:, start_idx:end_idx]

                # Perform update
                _loss, _, _hidden, _memory = sess.run(
                    [loss, train_op, hidden, memory],
                    feed_dict={
                        inputs_placeholder: batch_inputs,
                        labels_placeholder: batch_labels,
                        init_hidden: _hidden,
                        init_memory: _memory
                    })

                # Report status
                if batch_idx % 100 == 0:
                    print("batch: ", batch_idx, "xentropy", _loss)

if __name__ == "__main__":
    test()
