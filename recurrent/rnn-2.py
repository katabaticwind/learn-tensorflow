import tensorflow as tf
import numpy as np

# parameters
num_unroll = 50
batch_size = 64
test_batch_size = 1
hidden = 64
in_size = 67
out_size = 67

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

data, mappings, tokens = import_data('./data/', batch_size, num_unroll)



# define placeholders
train_inputs = []
for ui in range(num_unroll):
    train_inputs.append(tf.placeholder(tf.int32,
        shape=[batch_size, 1], name='train_inputs_%d'%ui))
train_labels = tf.placeholder(
    tf.int32,
    shape=[batch_size, num_unroll],
    name='train_labels_%d'%ui
)
train_one_hots = [tf.squeeze(tf.one_hot(pl, in_size), axis=1) for pl in train_inputs]

# define weights
W_xh = tf.Variable(tf.truncated_normal(
                   [in_size,hidden],stddev=0.02,
                   dtype=tf.float32),name='W_xh')
W_hh = tf.Variable(tf.truncated_normal([hidden,hidden],
                   stddev=0.02,
                   dtype=tf.float32),name='W_hh')
W_hy = tf.Variable(tf.truncated_normal(
                   [hidden,out_size],stddev=0.02,
                   dtype=tf.float32),name='W_hy')

# define state variables
prev_train_h = tf.Variable(
    tf.zeros([batch_size,hidden], dtype=tf.float32),
    name='train_h',
    trainable=False
)

# define outputs
outputs = list()
output_h = prev_train_h
for ui in range(num_unroll):
        output_h = tf.nn.tanh(
            tf.matmul(tf.concat([train_one_hots[ui],output_h],1),
                      tf.concat([W_xh,W_hh],0))
        )
        outputs.append(output_h)
y_scores = [tf.matmul(outputs[ui],W_hy) for ui in range(num_unroll)]
y_predictions = [tf.nn.softmax(y_scores[ui]) for ui in range(num_unroll)]

# define training operation
with tf.control_dependencies([tf.assign(prev_train_h, output_h)]):
    rnn_loss = tf.reduce_mean(
               tf.nn.sparse_softmax_cross_entropy_with_logits(
               logits=y_scores,
               labels=train_labels
    ))
rnn_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
gradients, v = zip(*rnn_optimizer.compute_gradients(rnn_loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
rnn_optimizer = rnn_optimizer.apply_gradients(zip(gradients, v))


# definte state reset operation
reset_train_h_op = tf.assign(prev_train_h,tf.zeros(
                             [batch_size,hidden],
                             dtype=tf.float32))

# train!
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for batch in range(len(data['inputs'])):
        if batch == 0:
            sess.run(reset_train_h_op)
        batch_inputs = np.split(np.squeeze(data['inputs'][batch], axis=1), num_unroll, axis=1)
        feed_dict = {i: d for i, d in zip(train_inputs, batch_inputs)}
        feed_dict[train_labels] = np.squeeze(data['labels'][batch], axis=1)
        xentropy, _ = sess.run([rnn_loss, rnn_optimizer], feed_dict=feed_dict)
        if batch % 1 == 0:
            print(f"batch={batch}, loss={xentropy:.2f}, perplexity={np.exp(xentropy):.2f}")
