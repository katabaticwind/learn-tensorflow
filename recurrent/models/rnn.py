from __future__ import print_function, division
import tensorflow as tf
import numpy as np


class RNN(object):
    """Basic RNN cell."""

    def __init__(self, input_dim, state_dim, output_dim, learning_rate):

        super(RNN, self).__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim

        self.Wxs = tf.Variable(
            tf.truncated_normal([input_dim, state_dim], stddev=0.02),
            dtype=tf.float32,
            name='weights_input'
        )
        self.Wss = tf.Variable(
            tf.truncated_normal([state_dim, state_dim], stddev=0.02),
            dtype=tf.float32,
            name='weights_state'
        )
        self.Wsy = tf.Variable(
            tf.truncated_normal([state_dim, output_dim], stddev=0.02),
            dtype=tf.float32,
            name='weights_output'
        )

        self.bs = tf.Variable(
            tf.constant(0.0, shape=[state_dim]),
            name='bias_state'
        )
        self.by = tf.Variable(
            tf.constant(0.0, shape=[output_dim]),
            name='bias_output'
        )

        self.lr = learning_rate

    def forward(self, inputs, init_state):
        """Construct graph for training RNN.

            # Arguments
            - `inputs`: list of [None, input_dim] placeholders
            - `state`: [None, state_dim] placeholder representing initial state

            # Returns
            - `logits`: list of [None, output_dim] tensors
            - `state`: [None, state_dim] tensor representing final state
        """
        state = init_state
        logits = []
        for input in inputs:
            state = tf.tanh(
                tf.add(
                    tf.add(
                        tf.matmul(input, self.Wxs),
                        tf.matmul(state , self.Wss)
                    ),
                    self.bs
                )
            )
            logits += [tf.add(tf.matmul(state, self.Wsy), self.by)]
        return logits, state

    def backward(self, logits, labels):
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )  # mapped across series -> [backprop_length, batch_size] tensor
        )
        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        return loss, train_op

    def predict(self, input, state, temperature=1.0):
        """Construct graph for predicting from an input.

            # Arguments
            - `input`: [None, input_dim] placeholder
            - `state`: [None, state_dim] placeholder representing initial state
            - `temperature`: scalar determining randomness of prediction

            # Returns
            - `prediction`: [None, output_dim] tensor representing predicted output
            - `state`: [None, state_dim] tensor representing next state
        """
        state = tf.nn.tanh(
            tf.add(
                tf.add(
                    tf.matmul(input, self.Wxs),
                    tf.matmul(state, self.Wss)
                ),
                self.bs
            )
        )
        logits = tf.add(tf.matmul(state / temperature, self.Wsy), self.by)
        prediction = tf.multinomial(logits, 1)
        return prediction, state


def test():

    num_epochs = 20
    total_series_length = 50000
    backprop_length = 15  # truncated backpropogation length (i.e. training sequence length)
    state_size = 4
    num_classes = 2
    echo_step = 3
    batch_size = 5
    batches_per_epoch = total_series_length // batch_size // backprop_length

    def generate_batches():
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
    inputs_series = [tf.reshape(i, [batch_size, 1]) for i in inputs_series]
    labels_series = tf.unstack(labels_placeholder, axis=1)

    rnn = RNN(1, state_size, num_classes, 1e-2)
    logits_series, state = rnn.forward(inputs_series, init_state)
    loss, train_op = rnn.backward(logits_series, labels_series)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_idx in range(num_epochs):

            print("New data, epoch", epoch_idx)

            # Generate new data
            x, y = generate_batches()

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
                _loss, _, _state = sess.run(
                    [loss, train_op, state],
                    feed_dict={
                        inputs_placeholder: batch_inputs,
                        labels_placeholder: batch_labels,
                        init_state: _state
                    })

                # Report status
                if batch_idx % 100 == 0:
                    print("batch: ", batch_idx, "xentropy", _loss)

if __name__ == "__main__":
    test()
