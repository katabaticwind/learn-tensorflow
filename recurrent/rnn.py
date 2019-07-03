import tensorflow as tf
import numpy as np

def import_data(dir, batch_size, sequence_len, dtype='int'):

    print('Loading text...')
    text = open(dir + 'iliad.txt').read()
    text += open(dir +'odyssey.txt').read()

    print('Creating training data...')
    # clean up text
    text = text.replace('\n\n', ' ')  # paragraph ends and section headings
    text = text.replace('\n', ' ')  # line ends

    # split text into characters
    chars = list(text)

    # create character mappings
    tokens = np.unique(chars)
    idx_to_char = {idx:char for (idx, char) in enumerate(tokens)}
    char_to_idx = {char:idx for (idx, char) in enumerate(tokens)}

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
    print('done.')

    data = {'inputs': inputs, 'labels': labels}
    mappings = {'idx_to_char': idx_to_char, 'char_to_idx': char_to_idx}

    return data, mappings, tokens

class SequenceData():
    pass

class RNN(object):
    """docstring for RNN."""

    def __init__(self, input_dim, state_dim, output_dim):

        super(RNN, self).__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim

        self.Wx = tf.Variable(
            tf.truncated_normal([input_dim, state_dim], stddev=0.02),
            dtype=tf.float32,
            name='weights_input'
        )
        self.Ws = tf.Variable(
            tf.truncated_normal([state_dim, state_dim], stddev=0.02),
            dtype=tf.float32,
            name='weights_state'
        )
        self.Wy = tf.Variable(
            tf.truncated_normal([state_dim, output_dim], stddev=0.02),
            dtype=tf.float32,
            name='weights_output'
        )

        # self.bs = tf.Variable(
        #     tf.constant(0.0, shape=[state_dim]),
        #     name='bias_state'
        # )
        # self.by = tf.Variable(
        #     tf.constant(0.0, shape=[output_dim]),
        #     name='bias_output'
        # )

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

    def inference(self, inputs):
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

    def predict(self, input, size=1, temperature=1.0):
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
        prediction = tf.multinomial(logits, size)
        return prediction

    def train(self, logits, labels, init_state):
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


input_dim = 67
state_dim = 64
output_dim = 67
seq_len = 64
batch_size = 64
lr = 1e-3
data, mappings, tokens = import_data('./data/', batch_size, seq_len)
rnn = RNN(input_dim, state_dim, output_dim)
inputs_pl = [tf.placeholder(tf.int32, [None, 1]) for _ in range(seq_len)]
labels_pl = tf.placeholder(tf.int32, [None, seq_len])
logits, init_state = rnn.inference(inputs_pl)
loss, train_op = rnn.train(logits, labels_pl, init_state)
reset_op, _ = rnn.reset()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while True:
        for batch in range(len(data['inputs'])):
            if batch == 0:
                sess.run(reset_op)
            batch_inputs = np.split(np.squeeze(data['inputs'][batch], axis=1), seq_len, axis=1)
            feed_dict = {i: d for i, d in zip(inputs_pl, batch_inputs)}
            feed_dict[labels_pl] = np.squeeze(data['labels'][batch], axis=1)
            xentropy, _ = sess.run([loss, train_op], feed_dict=feed_dict)
            if batch % 100 == 0:
                print(f"batch={batch}, loss={xentropy:.2f}, perplexity={np.exp(xentropy):.2f}")




# input_dim = 67
# state_dim = 32
# output_dim = 67
# seq_len = 8
# batch_size = 4
# inputs, labels, idx_to_char, char_to_idx, tokens = import_data('./data/', batch_size, seq_len)
# rnn = RNN(input_dim, state_dim, output_dim)
# inputs_train = [tf.placeholder(tf.int32, [None, 1]) for _ in range(seq_len)]
# inputs_train_onehot = [tf.squeeze(tf.one_hot(pl, input_dim), axis=1) for pl in inputs_train]
# init_state_train = tf.placeholder(tf.float32, [None, state_dim])
# logits, state_train = rnn.decode(inputs_train_onehot, init_state_train)
# inputs_test = tf.placeholder(tf.int32, [None, 1])
# inputs_test_onehot = tf.squeeze(tf.one_hot(inputs_test, input_dim), axis=1)
# init_state_test = tf.placeholder(tf.float32, [None, state_dim])
# prediction, state_test = rnn.predict(inputs_test_onehot, init_state_test)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     # train
#     batch_inputs = np.split(np.squeeze(inputs[0], axis=1), seq_len, axis=1)
#     feed_dict = {i: d for i, d in zip(inputs_train, batch_inputs)}
#     feed_dict[init_state_train] = np.zeros([batch_size, state_dim])
#     logits_np, state_np = sess.run([logits, state_train], feed_dict=feed_dict)
#
#     # test
#     init_input = np.array([char_to_idx[np.random.choice(tokens)]]).reshape(1, -1)
#     feed_dict = {inputs_test: init_input, init_state_test: np.zeros([1, state_dim])}
#     prediction_np, state_np = sess.run([prediction, state_test], feed_dict=feed_dict)
