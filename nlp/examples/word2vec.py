from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import zipfile
import math
import random
import numpy as np
import tensorflow as tf
import collections

from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'train', """'train' or 'test'.""")
tf.app.flags.DEFINE_string('data_dir', '../data/', """Directory where data is stored.""")
tf.app.flags.DEFINE_string('log_dir', './logs/', """Base directory for checkpoints and summaries.""")
tf.app.flags.DEFINE_integer('max_steps', 25000, """Number of epochs to train for.""")
tf.app.flags.DEFINE_integer('log_steps', 100, """Steps between logs.""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Observations per batch.""")
tf.app.flags.DEFINE_integer('num_words', 10000, """Size of dataset vocabulary.""")
tf.app.flags.DEFINE_integer('embedding_size', 256, """Size of embedding.""")
tf.app.flags.DEFINE_integer('skip_window', 1, """Index of context target.""")
tf.app.flags.DEFINE_integer('num_skips', 2, """Samples per context target.""")
tf.app.flags.DEFINE_integer('num_sampled', 64, """Size of NCE negative sample.""")
tf.app.flags.DEFINE_float('learning_rate', 1.0, """Learning rate.""")

def read_data(file):
    """Extract the first file enclosed in a zip file as a list of words."""
    print('reading data...')
    with zipfile.ZipFile(file) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        return data

def build_dataset(words, size):
    """Process raw inputs into a dataset.

    # Arguments
    - `words`: list of words in text corpus
    - `size`: number of words to include in the dataset

    # Returns
    - `data`: list of words represented as indices
    - `counts`: counts for words selected for dataset
    - `word2index`: mapping from word to index
    - `index2word`: mapping from index to word
    """
    print('building dataset...')
    counts = [['UNK', -1]]
    counts.extend(collections.Counter(words).most_common(size - 1))
    word2index = {word: index for index, (word, count) in enumerate(counts)}
    data = []
    unk_count = 0
    for word in words:
        index = word2index.get(word, 0)
        data.append(index)
        if index == 0:
            unk_count += 1
    counts[0][1] = unk_count
    index2word = {value: key for key,value in word2index.items()}
    print('Most common words (+UNK):', counts[:5])
    print('Sample data:', data[:10], [index2word[i] for i in data[:10]])
    return data, counts, word2index, index2word

class BatchGenerator(object):
    """docstring for BatchGenerator."""

    def __init__(self, data, batch_size, num_skips, skip_window):
        super(BatchGenerator, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.index = 0

    def next_batch(self):
        # un-modified data...
        batch_size = self.batch_size
        num_skips = self.num_skips
        skip_window = self.skip_window
        data = self.data
        assert batch_size % num_skips == 0  # e.g. batch_size = num_skips * 4
        assert num_skips <= 2 * skip_window  # at most use the whole context
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
        if self.index + span > len(data):
            self.index = 0
        buffer.extend(data[self.index:self.index + span])
        self.index += span
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
            if self.index == len(data):
                print("*finished epoch*")
                buffer.extend(data[0:span])
                self.index = span
            else:
                buffer.append(data[self.index])
                self.index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.index = (self.index + len(data) - span) % len(data)
        return batch, labels

def train(data_dir,
          log_dir,
          num_words,
          learning_rate,
          batch_size,
          embedding_size,
          skip_window,
          num_skips,
          num_sampled,
          max_steps,
          log_steps):

    vocabulary = read_data(data_dir + 'text8.zip')
    data, counts, word2index, index2word = build_dataset(vocabulary, num_words)

    # Construct graph

    # placeholders
    inputs_pl = tf.placeholder(tf.int32, shape=[batch_size])
    labels_pl = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # variables
    embeddings = tf.Variable(
        tf.random_uniform([num_words, embedding_size], -1.0, 1.0)
    )
    nce_weights = tf.Variable(
        tf.truncated_normal(
            [num_words, embedding_size],
            stddev=1.0 / math.sqrt(embedding_size)
        )
    )
    nce_biases = tf.Variable(tf.zeros([num_words]))

    # graph
    embedded_inputs = tf.nn.embedding_lookup(embeddings, inputs_pl)

    # loss
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=labels_pl,
            inputs=embedded_inputs,
            num_sampled=num_sampled,
            num_classes=num_words
        )
    )

    # update
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # summary
    tf.summary.scalar('loss', loss)
    summary_op = tf.summary.merge_all()

    # checkpoints
    saver = tf.train.Saver()

    # Train network
    generator = BatchGenerator(data, batch_size, num_skips, skip_window)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        for step in range(max_steps):

            # Get batch
            batch_inputs, batch_labels = generator.next_batch()

            # Run update
            feed_dict = {
                inputs_pl: batch_inputs,
                labels_pl: batch_labels
            }
            _, summary, nce = sess.run(
                [train_op, summary_op, loss],
                feed_dict=feed_dict
            )

            # Status report
            if step % log_steps == 0:
                print(f'step = {step}, loss = {nce}')

            # Write summaries
            writer.add_summary(summary, step)

        # Save the model for checkpoints.
        saver.save(sess, log_dir + '/model.ckpt')

        # Record embedding labels
        with open(log_dir + '/metadata.tsv', 'w') as f:
            for i in range(num_words):
                f.write(index2word[i] + '\n')

        # Visualization
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = log_dir + '/metadata.tsv'
        projector.visualize_embeddings(writer, config)

        writer.close()

    # Visualize results
    # TODO

if __name__ == '__main__':
    if FLAGS.mode == 'train':
        train(data_dir=FLAGS.data_dir,
              log_dir=FLAGS.log_dir,
              max_steps=FLAGS.max_steps,
              batch_size=FLAGS.batch_size,
              num_words=FLAGS.num_words ,
              embedding_size=FLAGS.embedding_size,
              skip_window=FLAGS.skip_window,
              num_skips=FLAGS.num_skips,
              num_sampled=FLAGS.num_sampled,
              learning_rate=FLAGS.learning_rate,
              log_steps=FLAGS.log_steps)
