import tensorflow as tf
import numpy as np
from datetime import datetime


def multinomial(outputs, state, temperature):
    outputs = tf.reshape(tf.pack(outputs, axis=1), [test_size, num_neurons])
    logits = tf.matmul(outputs / temperature, weights) + bias
    return tf.multinomial(logits, 1), state

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
    return inputs, labels, idx_to_char, char_to_idx, tokens


# config
dir = './data/'
log_dir = './logs/'
lr = 1e-2
max_epochs = 5
log_freq = 25
input_dim = 67
output_dim = 67
state_dim = 256
seq_len = 64
batch_size = 32
sample_size = 256

# network graph
weights_inputs = tf.Variable(
    tf.truncated_normal([input_dim, state_dim], stddev=0.02),
    dtype=tf.float32,
    name='weights_inputs'
)
weights_state = tf.Variable(
    tf.truncated_normal([state_dim, state_dim], stddev=0.02),
    dtype=tf.float32,
    name='weights_state'
)
weights_output = tf.Variable(
    tf.truncated_normal([state_dim, output_dim], stddev=0.02),
    dtype=tf.float32,
    name='weights_output'
)

bias_state = tf.Variable(
    tf.constant(0.0, shape=[state_dim]),
    name='bias_state'
)
bias_output = tf.Variable(
    tf.constant(0.0, shape=[output_dim]),
    name='bias_output'
)

# training graph
inputs_train = [tf.placeholder(dtype=tf.int32, shape=[None, 1]) for _ in range(seq_len)]  # *dense* inputs
state_train_pl = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])

def rnn_train(inputs, state):
    logits = []
    for input in inputs:
        input = tf.squeeze(tf.one_hot(input, output_dim), axis=1)
        state = tf.nn.tanh(
            tf.add(
                tf.add(
                    tf.matmul(input, weights_inputs),
                    tf.matmul(state, weights_state)
                ),
                bias_state
            )
        )
        logits += [
            tf.nn.relu(tf.add(tf.matmul(state, weights_output), bias_output))
        ]
    return logits, state

logits_train, state_train = rnn_train(inputs_train, state_train_pl)

labels_pl = tf.placeholder(shape=[None, seq_len], dtype=tf.int32)  # NOTE: *dense* labels
loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels_pl,
        logits=logits_train,
    ),
    name='loss',
)
opt = tf.train.AdamOptimizer(learning_rate=lr)
train_op = opt.minimize(loss)

# testing graph
input_test = tf.placeholder(dtype=tf.int32, shape=[None, 1])
state_test_pl = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])

def rnn_test(input, state):
    input = tf.squeeze(tf.one_hot(input, output_dim), axis=1)
    state = tf.nn.tanh(
        tf.add(
            tf.add(
                tf.matmul(input, weights_inputs),
                tf.matmul(state, weights_state)
            ),
            bias_state
        )
    )
    logits = tf.nn.relu(tf.add(tf.matmul(state, weights_output), bias_output))
    prediction = tf.multinomial(logits, 1)
    return prediction, state

prediction, state_test = rnn_test(input_test, state_test_pl)

# summary graph
tf.summary.scalar('cross_entropy', loss)
tf.summary.histogram('weights_inputs', weights_inputs)
tf.summary.histogram('weights_state', weights_state)
tf.summary.histogram('weights_output', weights_output)
tf.summary.histogram('logits', logits_train)
tf.summary.histogram('state', state_train)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

# load data
inputs, labels, idx_to_char, char_to_idx, tokens = import_data(dir, batch_size, seq_len)

# train network
def train():
    with tf.Session() as sess:
        sess.run(init_op)
        now = datetime.today()
        date_string = now.strftime("%Y-%m-%d-%H:%M:%S.%f")
        writer = tf.summary.FileWriter(log_dir + date_string, sess.graph)
        global_step = 0
        for epoch in range(max_epochs):
            for batch in range(len(inputs)):
                batch_inputs = np.split(inputs[batch].squeeze(axis=1), seq_len, axis=1)
                batch_labels = labels[batch].squeeze(axis=1)
                if batch == 0:
                    state_np = np.zeros([batch_size, state_dim])  # reset state_np
                feed_dict = {i: d for i, d in zip(inputs_train, batch_inputs)}
                feed_dict[state_train_pl] = state_np
                feed_dict[labels_pl] = batch_labels
                # perform an update
                if batch % log_freq == 0:
                    out = sess.run([state_train, loss, train_op, summary_op], feed_dict=feed_dict)
                    state_np, xentropy_loss, _, summary = out
                    writer.add_summary(summary, global_step)
                    print(f"epoch={epoch}, batch={batch}, xentropy={xentropy_loss:.2f}")
                    # inputs_seq = [batch_inputs[i][0] for i in range(seq_len)]
                    # print(''.join([idx_to_char[i[0]] for i in inputs_seq]))
                    # labels_seq = batch_labels[0, :].tolist()
                    # print(''.join([idx_to_char[i] for i in labels_seq]))
                else:
                    out = sess.run([state_train, loss, train_op], feed_dict=feed_dict)
                    state_np, xentropy_loss, _ = out
                global_step += 1

def test():
    with tf.Session() as sess:
        sess.run(init_op)
        idx = char_to_idx[np.random.choice(tokens)]
        state_np = np.zeros([1, state_dim])  # initial state
        output = [idx_to_char[idx]]
        idx = np.array([[idx]])
        while len(output) < sample_size:
            idx, state_np = sess.run(
                [prediction, state_test],
                feed_dict={
                    input_test: idx,
                    state_test_pl: state_np,
                })
            output += [idx_to_char[idx[0, 0]]]

    text = ''.join(output)
    print(text)
    return text

# w/ class

if __name__ == '__main__':
    train()
    test()
