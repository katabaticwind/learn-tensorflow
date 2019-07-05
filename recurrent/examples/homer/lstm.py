from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import time
import os
from datetime import datetime

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('mode', 'train', """'train' or 'test'.""")
tf.app.flags.DEFINE_string('data_dir', '../../data/', """Directory where data is stored.""")
tf.app.flags.DEFINE_string('base_dir', '.', """Base directory for checkpoints and summaries.""")
tf.app.flags.DEFINE_integer('num_epochs', 100, """Number of epochs to train for.""")
tf.app.flags.DEFINE_integer('batch_size', 64, """Sequences per batch.""")
tf.app.flags.DEFINE_integer('backprop_length', 64, """Training sequence length.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """Learning rate.""")
tf.app.flags.DEFINE_integer('input_dim', 67, """Size of input.""")
tf.app.flags.DEFINE_integer('output_dim', 67, """Size of output.""")
tf.app.flags.DEFINE_integer('hidden_dim', 512, """Size of hidden state.""")
tf.app.flags.DEFINE_integer('sample_len', 512, """Size of text sample(s).""")
tf.app.flags.DEFINE_float('temp', .2, """Temperature of softmax during text generation.""")


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
                outputs += [tf.nn.dropout(hidden, keep_prob)]
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
        next_hidden = tf.multiply(o, tf.tanh(next_memory))
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

    def predict(self, input, hiddens, memories):
        hidden_list = []
        memory_list = []
        for (cell, hidden, memory) in zip(self.cells, hiddens, memories):
            hidden, memory = cell.predict(input, hidden, memory)
            input = output = hidden
            hidden_list += [hidden]
            memory_list += [memory]
        return output, hidden_list, memory_list

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

def random_input(tokens, char_to_idx):
    char = np.random.choice(tokens)
    idx = char_to_idx[char]
    return np.array([[idx]])  # placeholder is [None, 1]

def create_batches(character_list, batch_size, backprop_length):
    """
        Create list of arrays with shape `[batch_size, backprop_length]`.

        # Argument
        -  `character_list`: [batch_size, ...] array
    """
    batch_size, total_length = character_list.shape
    sections, remainder = divmod(total_length, backprop_length)
    return np.split(character_list[:, :-remainder], sections, axis=-1)

def split_inputs(inputs, dim, backprop_length):
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

def split_labels(labels, backprop_length):
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

def multinomial(input, weights, bias, temperature=1.0):
    """
        `temperature` -> 0 => argmax(logits)
    """
    logits = tf.matmul(input, weights) + bias
    return tf.multinomial(logits / temperature, 1)
    # logits = tf.matmul(input / temperature, weights) + bias
    # return tf.multinomial(logits, 1)

def backward(logits, labels, learning_rate):
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels
        )
    )
    # opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)
    return loss, train_op

def create_directories(base_dir = "."):
    """Create a directories to save logs and checkpoints.

        E.g. ./checkpoints/5-14-2019-9:37:25.43/
    """

    now = datetime.today()
    date_string = now.strftime("%Y-%m-%d-%H:%M:%S.%f")
    ckpt_dir = "/".join([base_dir, "checkpoints", date_string])
    log_dir = "/".join([base_dir, "logs", date_string])
    meta_dir = "/".join([base_dir, "meta", date_string])
    os.makedirs(ckpt_dir)
    os.makedirs(log_dir)
    os.makedirs(meta_dir)
    return ckpt_dir, log_dir, meta_dir

def find_latest_checkpoint(load_path, prefix):
    """Find the latest checkpoint in dir at `load_path` with prefix `prefix`

        E.g. ./checkpoints/dqn-vanilla-CartPole-v0-GLOBAL_STEP would use find_latest_checkpoint('./checkpoints/', 'dqn-vanilla-CartPole-v0')
    """
    files = os.listdir(load_path)
    matches = [f for f in files if f.find(prefix) == 0]  # files starting with prefix
    max_steps = np.max(np.unique([int(m.strip(prefix).split('.')[0]) for m in matches]))
    latest_checkpoint = load_path + prefix + str(max_steps)
    return latest_checkpoint

def train(data_dir='../../data/',
          base_dir='.',
          num_epochs=100,
          batch_size=64,
          backprop_length=64,
          input_dim=67,
          output_dim=67,
          hidden_dim=512,
          temp=0.5,
          learning_rate=1e-3,
          sample_len=128):

    # Setup
    ckpt_dir, log_dir, meta_dir = create_directories(base_dir)

    # Import data
    data, mappings, tokens = import_data(data_dir, batch_size, backprop_length)
    inputs_batches = data['inputs']
    labels_batches = data['labels']
    idx_to_char = mappings['idx_to_char']
    char_to_idx = mappings['char_to_idx']
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
    inputs_series = split_inputs(inputs_placeholder, output_dim, backprop_length)
    labels_series = split_labels(labels_placeholder, backprop_length)
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

    # Summary operations
    tf.summary.scalar('xentropy', loss)
    summary_op = tf.summary.merge_all()

    # Text generation graph
    input_pl = tf.placeholder(tf.int32, [None, 1])
    input_one_hot = tf.squeeze(tf.one_hot(input_pl, input_dim), axis=1)
    hidden_pls = [
        tf.placeholder(tf.float32, [None, hidden_dim]),
        tf.placeholder(tf.float32, [None, hidden_dim])
    ]
    memory_pls = [
        tf.placeholder(tf.float32, [None, hidden_dim]),
        tf.placeholder(tf.float32, [None, hidden_dim])
    ]
    output_predict, next_hidden, next_memory = lstm.predict(
        input_one_hot,
        hidden_pls,
        memory_pls
    )
    prediction = multinomial(output_predict, Why, by, temperature=temp)

    # Checkpoint saver
    saver = tf.train.Saver()

    # Loop
    print("Beginning training...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        global_step = 0

        def generate_sample(sample_len):
            print("Generating sample text...")
            # Choose a random input
            input_np = random_input(tokens, char_to_idx)

            # Set initial states
            hidden_np = [
                np.zeros((batch_size, hidden_dim)),
                np.zeros((batch_size, hidden_dim))
            ]
            memory_np = [
                np.zeros((batch_size, hidden_dim)),
                np.zeros((batch_size, hidden_dim))
            ]

            # Create feed dict
            feed_dict = {}
            feed_dict[hidden_pls[0]] = hidden_np[0]
            feed_dict[hidden_pls[1]] = hidden_np[1]
            feed_dict[memory_pls[0]] = memory_np[0]
            feed_dict[memory_pls[1]] = memory_np[1]
            feed_dict[input_pl] = input_np

            # generate sample...
            sample = [idx_to_char[input_np[0, 0]]]
            while len(sample) < sample_len:
                input_np, hidden_np, memory_np = sess.run(
                    [prediction, next_hidden, next_memory],
                    feed_dict=feed_dict
                )
                sample += [idx_to_char[input_np[0, 0]]]
                feed_dict = {}
                feed_dict[hidden_pls[0]] = hidden_np[0]
                feed_dict[hidden_pls[1]] = hidden_np[1]
                feed_dict[memory_pls[0]] = memory_np[0]
                feed_dict[memory_pls[1]] = memory_np[1]
                feed_dict[input_pl] = input_np

            print(''.join(sample))

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

                start_time = time.time()

                # get batch data
                batch_inputs = inputs_batches[batch_idx]
                batch_labels = labels_batches[batch_idx]

                # create feed dictionary
                feed_dict = {}
                feed_dict[init_hidden[0]] = _hidden[0]
                feed_dict[init_hidden[1]] = _hidden[1]
                feed_dict[init_memory[0]] = _memory[0]
                feed_dict[init_memory[1]] = _memory[1]
                feed_dict[inputs_placeholder] = batch_inputs
                feed_dict[labels_placeholder] = batch_labels
                feed_dict[keep_pl] = 0.5

                # perform update
                if batch_idx % 10 == 0:  # w/ summary
                    _loss, _, _hidden, _memory, _summary = sess.run(
                        [loss, train_op, hidden, memory, summary_op],
                        feed_dict=feed_dict
                    )
                    writer.add_summary(_summary, global_step)
                    batch_time = time.time() - start_time
                    print("batch: ", batch_idx, "xentropy: ", _loss, "seconds: ", batch_time)
                else:  # w/o summary
                    _loss, _, _hidden, _memory = sess.run(
                        [loss, train_op, hidden, memory],
                        feed_dict=feed_dict
                    )
                global_step += 1

                # save checkpoint
                if batch_idx % 100 == 0:
                    saver.save(sess, save_path=ckpt_dir + "/ckpt", global_step=global_step)
                    generate_sample(sample_len)

def test(data_dir='../../data/',
         base_dir='.',
         batch_size=64,
         backprop_length=64,
         input_dim=67,
         output_dim=67,
         hidden_dim=256,
         temp=0.5,
         sample_len=512):

    # e.g. base_dir = './checkpoints/2019-07-04-17:33:42.726068/'
    print(base_dir)
    restore_dir = find_latest_checkpoint(base_dir, 'ckpt-')
    print(restore_dir)

    # import data (to get tokens)
    data, mappings, tokens = import_data(data_dir, batch_size, backprop_length)
    inputs_batches = data['inputs']
    labels_batches = data['labels']
    idx_to_char = mappings['idx_to_char']
    char_to_idx = mappings['char_to_idx']
    batches_per_epoch = len(inputs_batches)

    # print("Defining placeholders...")
    # inputs_placeholder = tf.placeholder(tf.int32, [batch_size, backprop_length])
    # labels_placeholder = tf.placeholder(tf.int32, [batch_size, backprop_length])
    # init_hidden = [
    #     tf.placeholder(tf.float32, [batch_size, hidden_dim]),
    #     tf.placeholder(tf.float32, [batch_size, hidden_dim])
    # ]
    # init_memory = [
    #     tf.placeholder(tf.float32, [batch_size, hidden_dim]),
    #     tf.placeholder(tf.float32, [batch_size, hidden_dim])
    # ]
    # inputs_series = split_inputs(inputs_placeholder, output_dim, backprop_length)
    # labels_series = split_labels(labels_placeholder)
    # keep_pl = tf.placeholder(tf.float32)

    print("Re-constructing graph...")
    lstm_1 = LSTM(input_dim=input_dim, hidden_dim=hidden_dim)
    lstm_2 = LSTM(input_dim=hidden_dim, hidden_dim=hidden_dim)
    lstm = Chain([lstm_1, lstm_2])
    # outputs_series, hidden, memory = lstm.forward(
    #     inputs_series,
    #     init_hidden,
    #     init_memory,
    #     keep_pl
    # )
    Why = tf.Variable(
        tf.truncated_normal([hidden_dim, output_dim], stddev=0.02),
        dtype=tf.float32,
        name='weights_softmax'
    )
    by = tf.Variable(
        tf.constant(0.0, shape=[output_dim]),
        name='bias_softmax'
    )
    # logits_series = [tf.matmul(h, Why) + by for h in outputs_series]
    # loss, train_op = backward(logits_series, labels_series, learning_rate)

    # Text generation graph
    input_pl = tf.placeholder(tf.int32, [None, 1])
    input_one_hot = tf.squeeze(tf.one_hot(input_pl, input_dim), axis=1)
    hidden_pls = [
        tf.placeholder(tf.float32, [None, hidden_dim]),
        tf.placeholder(tf.float32, [None, hidden_dim])
    ]
    memory_pls = [
        tf.placeholder(tf.float32, [None, hidden_dim]),
        tf.placeholder(tf.float32, [None, hidden_dim])
    ]
    output_predict, next_hidden, next_memory = lstm.predict(
        input_one_hot,
        hidden_pls,
        memory_pls
    )
    prediction = multinomial(output_predict, Why, by, temperature=temp)

    # Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, restore_dir)

        print("Generating sample text...")
        # Choose a random input
        input_np = random_input(tokens, char_to_idx)

        # Set initial states
        hidden_np = [
            np.zeros((batch_size, hidden_dim)),
            np.zeros((batch_size, hidden_dim))
        ]
        memory_np = [
            np.zeros((batch_size, hidden_dim)),
            np.zeros((batch_size, hidden_dim))
        ]

        # Create feed dict
        feed_dict = {}
        feed_dict[hidden_pls[0]] = hidden_np[0]
        feed_dict[hidden_pls[1]] = hidden_np[1]
        feed_dict[memory_pls[0]] = memory_np[0]
        feed_dict[memory_pls[1]] = memory_np[1]
        feed_dict[input_pl] = input_np

        # generate sample...
        sample = [idx_to_char[input_np[0, 0]]]
        while len(sample) < sample_len:
            input_np, hidden_np, memory_np = sess.run(
                [prediction, next_hidden, next_memory],
                feed_dict=feed_dict
            )
            sample += [idx_to_char[input_np[0, 0]]]
            feed_dict = {}
            feed_dict[hidden_pls[0]] = hidden_np[0]
            feed_dict[hidden_pls[1]] = hidden_np[1]
            feed_dict[memory_pls[0]] = memory_np[0]
            feed_dict[memory_pls[1]] = memory_np[1]
            feed_dict[input_pl] = input_np

    print(''.join(sample))


if __name__ == '__main__':
    if FLAGS.mode == 'train':
        train(data_dir=FLAGS.data_dir,
              base_dir=FLAGS.base_dir,
              num_epochs=FLAGS.num_epochs,
              batch_size=FLAGS.batch_size,
              backprop_length=FLAGS.backprop_length ,
              input_dim=FLAGS.input_dim,
              output_dim=FLAGS.output_dim,
              hidden_dim=FLAGS.hidden_dim,
              learning_rate=FLAGS.learning_rate,
              sample_len=FLAGS.sample_len)
    elif FLAGS.mode == 'test':
        test(data_dir=FLAGS.data_dir,
             base_dir=FLAGS.base_dir,
             batch_size=FLAGS.batch_size,
             backprop_length=FLAGS.backprop_length ,
             input_dim=FLAGS.input_dim,
             output_dim=FLAGS.output_dim,
             hidden_dim=FLAGS.hidden_dim,
             temp=FLAGS.temp,
             sample_len=FLAGS.sample_len)
