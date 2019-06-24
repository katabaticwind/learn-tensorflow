import tensorflow as tf
import numpy as np

# TODO: Add a layer on top of the RNN layer to convert output to proper logits. The current output is determined by the number of hidden units in the RNN cell, but it should be determined by the size of the desired output (e.g. the number of classes available).

"""Example of an RNN cell."""
def rnn(inputs, state, hidden_units, activation):
    """Create a basic RNN unit.

        s[t] = tanh(x[t] * W_x + s[t - 1] * W_s)
        y[t] = activation(s[t] * W_y)

        # Arguments
        - `inputs::Tensor`: tensor with shape [None, inputs_dim]
        - `state::Tensor`: tensor with shape [None, state_dim]
        - `hidden_units::Int`: number of hidden units.
        - `activation`: activation to apply to hidden layer.

        # Returns
        - `state::Tensor`: tensor with shape [None, state_dim]
        - `output::Tensor`: tensor with shape [None, hidden_units]
    """

    inputs_dim = inputs.shape[1].value
    state_dim = state.shape[1].value

    weights_inputs = tf.Variable(
        tf.truncated_normal([inputs_dim, state_dim], stddev=0.02),
        dtype=tf.float32,
        name='weights_inputs'
    )
    weights_state = tf.Variable(
        tf.truncated_normal([state_dim, state_dim], stddev=0.02),
        dtype=tf.float32,
        name='weights_state'
    )
    weights_output = tf.Variable(
        tf.truncated_normal([state_dim, hidden_units], stddev=0.02),
        dtype=tf.float32,
        name='weights_output'
    )

    bias_inputs = tf.Variable(
        tf.constant(0.0, shape=[state_dim]),
        name='bias_inputs'
    )
    bias_state = tf.Variable(
        tf.constant(0.0, shape=[state_dim]),
        name='bias_state'
    )
    bias_output = tf.Variable(
        tf.constant(0.0, shape=[hidden_units]),
        name='bias_output'
    )

    state = tf.nn.tanh(
        tf.add(
            tf.add(tf.matmul(inputs, weights_inputs), bias_inputs),
            tf.add(tf.matmul(state, weights_state), bias_state)
        )
    )

    output = activation(
        tf.add(tf.matmul(state, weights_output), bias_output)
    )

    return state, output

def rnn_example():
    inputs_pl = tf.placeholder(shape=[None, 10], dtype=tf.float32)
    state_pl = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    outputs = rnn(inputs_pl, state_pl, hidden_units=16, activation=tf.nn.relu)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        feed_dict = {
            inputs_pl: np.random.randn(32, 10),
            state_pl: np.zeros([32, 2])
        }
        next_state, logits = sess.run(outputs, feed_dict=feed_dict)


"""Examples of RNN networks."""
def rnn_unrolled(inputs, state, hidden_units, activation, steps):

    inputs_dim = inputs.shape[1].value
    state_dim = state.shape[1].value

    weights_inputs = tf.Variable(
        tf.truncated_normal([inputs_dim, state_dim], stddev=0.02),
        dtype=tf.float32,
        name='weights_inputs'
    )
    weights_state = tf.Variable(
        tf.truncated_normal([state_dim, state_dim], stddev=0.02),
        dtype=tf.float32,
        name='weights_state'
    )
    weights_output = tf.Variable(
        tf.truncated_normal([state_dim, hidden_units], stddev=0.02),
        dtype=tf.float32,
        name='weights_output'
    )

    bias_inputs = tf.Variable(
        tf.constant(0.0, shape=[state_dim]),
        name='bias_inputs'
    )
    bias_state = tf.Variable(
        tf.constant(0.0, shape=[state_dim]),
        name='bias_state'
    )
    bias_output = tf.Variable(
        tf.constant(0.0, shape=[hidden_units]),
        name='bias_output'
    )

    logits = []
    for i in range(steps):
        state = tf.nn.tanh(
            tf.add(
                tf.add(tf.matmul(inputs[:, :, i], weights_inputs), bias_inputs),
                tf.add(tf.matmul(state, weights_state), bias_state)
            )
        )
        logits += [
            activation(tf.add(tf.matmul(state, weights_output), bias_output))
        ]

    return state, tf.stack(logits, axis=1)

def rnn_unrolled_example():
    steps = 3
    inputs_pl = tf.placeholder(shape=[None, 10, steps], dtype=tf.float32)
    state_pl = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    outputs = rnn_unrolled(
        inputs_pl,
        state_pl,
        hidden_units=16,
        activation=tf.nn.relu,
        steps=steps
    )
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        feed_dict = {
            inputs_pl: np.random.randn(32, 10, steps),
            state_pl: np.zeros([32, 2])
        }
        final_state, logits = sess.run(outputs, feed_dict=feed_dict)

def rnn_unrolled_training_example():
    lr = 1e-3
    max_updates = 10
    steps = 10
    inputs_dim = 5
    state_dim = 2
    hidden_units = 16
    inputs_pl = tf.placeholder(shape=[None, inputs_dim, steps], dtype=tf.float32)
    state_pl = tf.placeholder(shape=[None, state_dim], dtype=tf.float32)
    state, logits = rnn_unrolled(
        inputs_pl,
        state_pl,
        hidden_units=hidden_units,
        activation=tf.nn.relu,
        steps=steps
    )
    targets_pl = tf.placeholder(shape=[None, steps], dtype=tf.int32)  # sparse labels
    cross_entropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets_pl,
        logits=logits,
        name='cross_entropy'
    )
    loss = tf.reduce_mean(cross_entropy_losses, name='loss')
    opt = tf.train.RMSPropOptimizer(learning_rate=lr)
    train_op = opt.minimize(loss)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        state_np = np.zeros([32, state_dim])  # initial state
        feed_dict = {
            inputs_pl: np.random.randn(32, inputs_dim, steps),
            state_pl: state_np,
            targets_pl: np.random.randint(low=0, high=hidden_units, size=[32, steps])
        }
        for _ in range(max_updates):
            # perform an update
            state_np, logits_np, loss_np, _ = sess.run(
                [state, logits, loss, train_op],
                feed_dict=feed_dict
            )
            print(f"{loss_np}")
            # update feed_dict
            feed_dict = {
                inputs_pl: np.random.randn(32, inputs_dim, steps),
                state_pl: state_np,  # final state from previous batch
                targets_pl: np.random.randint(low=0, high=hidden_units, size=[32, steps])
            }
