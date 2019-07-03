import tensorflow as tf
import numpy as np

class RNN(object):
    """docstring for RNN."""

    def __init__(self, input_dim, state_dim, output_dim):

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

        self.state_train = tf.Variable(
            tf.zeros([batch_size, state_dim], dtype=tf.float32),
            name='state_train',
            trainable=False
        )
        self.state_valid = tf.Variable(
            tf.zeros([batch_size, state_dim], dtype=tf.float32),
            name='state_valid',
            trainable=False
        )

    def forward(self, inputs):
        """Construct graph for training RNN.

            # Arguments
            - `inputs`: list of [None, input_dim] placeholders
            - `state`: [None, state_dim] placeholder representing initial state

            # Returns
            - `logits`: list of [None, output_dim] tensors
            - `state`: [None, state_dim] tensor representing final state
        """
        logits = []
        state = self.state_train
        inputs = self.one_hot_inputs(inputs)
        for input in inputs:
            state = tf.nn.tanh(
                tf.add(
                    tf.matmul(input, self.Wx),
                    tf.matmul(state, self.Ws)
                )
            )
            logits += [tf.matmul(state, self.Wy)]
        return logits, state

    def backward(self, logits, labels, init_state):
        with tf.control_dependencies([tf.assign(self.state_train, init_state)]):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=labels
                )
            )
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = opt.minimize(loss)
        return loss, train_op

    def predict(self, input, temperature=1.0):
        """Construct graph for predicting from an input.

            # Arguments
            - `input`: [None, input_dim] placeholder
            - `state`: [None, state_dim] placeholder representing initial state
            - `temperature`: scalar determining randomness of prediction

            # Returns
            - `prediction`: [None, output_dim] tensor representing predicted output
            - `state`: [None, state_dim] tensor representing next state
        """
        state = self.state_test
        state = tf.nn.tanh(
            tf.add(
                tf.matmul(input, self.Wx),
                tf.matmul(state, self.Ws)
            )
        )
        logits = tf.matmul(state / temperature, self.Wy)
        prediction = tf.multinomial(logits, 1)
        return prediction



    def one_hot_inputs(self, placeholders):
        return [tf.squeeze(tf.one_hot(pl, self.input_dim), axis=1) for pl in placeholders]

    def reset(self):
        reset_train_op = tf.assign(
            self.state_train,
            tf.zeros([batch_size, self.state_dim], dtype=tf.float32)
        )
        reset_test_op = tf.assign(
            self.state_valid,
            tf.zeros([batch_size, self.state_dim], dtype=tf.float32)
        )
        return reset_train_op, reset_test_op

def test():
