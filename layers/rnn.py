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
def rnn_unrolled(inputs, state, hidden_units, activation, sequence_len):
    """Create an one-to-one "unrolled" RNN network.

        # Arguments
        - `inputs (tensor)`: stacked inputs with shape [None, input_dim, sequence_len]
        - `state (tensor)`: [None, state_dim]
        - `hidden_units (int)`: number of hidden units in the output layer
        - `activation (func)`: activation of the output layer
        - `sequence_len (int)`: number of "unrolling" steps

        # Outputs
        - `state (tensor)`: final hidden state
        - `logits (tensor)`: sequence of logits with shape [None, hidden_units, sequence_len]
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

    logits = []
    for i in range(sequence_len):
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
    sequence_len = 3
    inputs_pl = tf.placeholder(shape=[None, 10, sequence_len], dtype=tf.float32)
    state_pl = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    outputs = rnn_unrolled(
        inputs_pl,
        state_pl,
        hidden_units=16,
        activation=tf.nn.relu,
        sequence_len=sequence_len
    )
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        feed_dict = {
            inputs_pl: np.random.randn(32, 10, sequence_len),
            state_pl: np.zeros([32, 2])
        }
        final_state, logits = sess.run(outputs, feed_dict=feed_dict)

def rnn_unrolled_char_example():
    # set parameters
    lr = 1e-3
    max_updates = 1000
    batch_size = 32
    sequence_len = 64
    inputs_dim = 1
    state_dim = 32
    hidden_units = 65  # number of unique characters
    # construct graph
    inputs_pl = tf.placeholder(shape=[None, inputs_dim, sequence_len], dtype=tf.float32)
    state_pl = tf.placeholder(shape=[None, state_dim], dtype=tf.float32)
    state, logits = rnn_unrolled(
        inputs_pl,
        state_pl,
        hidden_units=hidden_units,
        activation=tf.nn.relu,
        sequence_len=sequence_len
    )
    labels_pl = tf.placeholder(shape=[None, sequence_len], dtype=tf.int32)  # sparse labels
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_pl,
            logits=logits,
        ),
        name='loss',
    )
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = opt.minimize(loss)
    init_op = tf.global_variables_initializer()
    # load data
    txt = open('./data/iliad.txt').read()
    inputs, labels, idx_to_char, _ = preprocess_text(txt, batch_size, sequence_len)
    # train network
    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(max_epochs):
            for step in range(max_updates):
                if step == 0:
                    feed_dict = {
                        inputs_pl: inputs[0].astype(np.float32),
                        state_pl: np.zeros([batch_size, state_dim]),
                        labels_pl: np.squeeze(labels[0], axis=1)
                    }
                else:
                    feed_dict = {
                        inputs_pl: inputs[step].astype(np.float32),
                        state_pl: state_np,  # final state from previous batch
                        labels_pl: np.squeeze(labels[step], axis=1)
                    }
                # perform an update
                state_np, cross_entropy_loss, _ = sess.run([state, loss, train_op], feed_dict=feed_dict)
                print(f"epoch={epoch}, step={step}, loss={cross_entropy_loss}")

def rnn_unrolled_hmm_example():
    # set parameters
    lr = 1e-4
    max_updates = 10000
    batch_size = 32
    sequence_len = 8
    inputs_dim = 1
    state_dim = 32
    hidden_units = 2
    # construct graph
    inputs_pl = tf.placeholder(shape=[None, inputs_dim, sequence_len], dtype=tf.float32)
    state_pl = tf.placeholder(shape=[None, state_dim], dtype=tf.float32)
    state, logits = rnn_unrolled(
        inputs_pl,
        state_pl,
        hidden_units=hidden_units,
        activation=tf.nn.relu,
        sequence_len=sequence_len
    )
    targets_pl = tf.placeholder(shape=[None, sequence_len], dtype=tf.int32)  # sparse labels
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets_pl,
            logits=logits,
        ),
        name='loss',
    )
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = opt.minimize(loss)
    init_op = tf.global_variables_initializer()
    # generate data
    X, Y = generate_data(sequence_len * batch_size * max_updates)
    X, Y = preprocess_data(X, Y, batch_size, sequence_len)
    # train network
    with tf.Session() as sess:
        sess.run(init_op)
        state_np = np.zeros([batch_size, state_dim])  # initialize state
        idx = np.random.randint(len(X))
        feed_dict = {
            inputs_pl: np.squeeze(X[idx], axis=-1),
            state_pl: state_np,
            targets_pl: np.squeeze(Y[idx]),
        }
        for step in range(max_updates):
            # perform an update
            cross_entropy_loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)
            print(f"step={step}, loss={cross_entropy_loss}")
            # update feed_dict
            idx = np.random.randint(len(X))
            feed_dict = {
                inputs_pl: np.squeeze(X[idx], axis=-1),
                state_pl: state_np,  # final state from previous batch
                targets_pl: np.squeeze(Y[idx]),
            }

def generate_data(n):
    X = np.random.randn(n)
    Y = []
    H = [np.random.binomial(1, 0.5)]
    A = np.random.uniform(size=[2, 2])
    A = A / np.sum(A, axis=-1, keepdims=True)
    print(A)
    P = [0.1, 0.9]  # [P(y = 1 | h = 0), P(y = 1 | h = 1)]
    Y += [np.random.binomial(1, P[H[0]])]
    for i in range(1, n):
        p = A[H[i - 1], 1]
        H += [np.random.binomial(1, p)]
        p = P[H[i]]
        Y += [np.random.binomial(1, p)]
    return X, np.array(Y)

def preprocess_data(X, Y, batch_size, sequence_len):
    # split data into len // batch_size
    n = len(X)
    obs_per_segment = n // batch_size
    if n % batch_size > 0:
        X = X[:-(n % batch_size)].reshape(batch_size, -1)
        Y = Y[:-(n % batch_size)].reshape(batch_size, -1)
    else:
        X = X.reshape(batch_size, -1)
        Y = Y.reshape(batch_size, -1)
    sequences_per_segment = obs_per_segment // sequence_len
    if obs_per_segment % sequence_len > 0:
        X = X[:, :-(obs_per_segment % sequence_len)].reshape(batch_size, 1, sequence_len, -1)
        Y = Y[:, :-(obs_per_segment % sequence_len)].reshape(batch_size, 1, sequence_len, -1)
    else:
        X = X.reshape(batch_size, 1, sequence_len, -1)
        Y = Y.reshape(batch_size, 1, sequence_len, -1)
    X = np.split(X, sequences_per_segment, axis=-1)  # batch_size x 1 x sequence_len x 1
    Y = np.split(Y, sequences_per_segment, axis=-1)  # batch_size x 1 x sequence_len x 1
    return X, Y

def preprocess_text(txt, batch_size, sequence_len, dtype='int'):

    # clean up text
    txt = txt.replace('\n\n', ' ')  # paragraph ends and section headings
    txt = txt.replace('\n', ' ')  # line ends

    # split text into characters
    chars = list(txt)

    # create character mappings
    idx_to_char = {idx:char for (idx, char) in enumerate(np.unique(chars))}
    char_to_idx = {char:idx for (idx, char) in enumerate(np.unique(chars))}

    if dtype == 'int':
        # convert characters to ints
        idx = [char_to_idx[c] for c in chars]

        # split integers into inputs and label
        inputs = np.array(idx[:-1])
        labels = np.array(idx[1:])

    else:
        # split characters into inputs and labels
        inputs = np.array(chars[:-1])
        labels = np.array(chars[1:])

    # create batches and segments
    n = len(inputs)
    chars_per_segment, r = divmod(n, batch_size)
    if r == 0:
        inputs = inputs.reshape(batch_size, -1)
        labels = labels.reshape(batch_size, -1)
    else:
        inputs = inputs[:-r].reshape(batch_size, -1)
        labels = labels[:-r].reshape(batch_size, -1)

    # divide segments into sequences
    sequences_per_segment, r = divmod(chars_per_segment, sequence_len)
    if r == 0:
        inputs = inputs.reshape(batch_size, -1, sequence_len).transpose([0, 2, 1])
        labels = labels.reshape(batch_size, -1, sequence_len).transpose([0, 2, 1])
    else:
        inputs = inputs[:, :-r].reshape(batch_size, -1, sequence_len)
        labels = labels[:, :-r].reshape(batch_size, -1, sequence_len)
    inputs = np.split(inputs, sequences_per_segment, axis=1)
    labels = np.split(labels, sequences_per_segment, axis=1)
    return inputs, labels, idx_to_char, char_to_idx
