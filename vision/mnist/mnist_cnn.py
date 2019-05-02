import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'viz', "Run script in training or testing mode.")

def load_data():
    print("Loading data...")
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    images_train = mnist.train.images
    labels_train = np.asarray(mnist.train.labels, dtype=np.int32)
    images_test = mnist.test.images
    labels_test = np.asarray(mnist.test.labels, dtype=np.int32)
    return images_train, labels_train, images_test, labels_test

def cnn(inputs):

    # conv1
    inputs = tf.reshape(inputs, (-1, 28, 28, 1))
    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=2
    )

    # conv2
    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=2
    )

    # dense
    inputs = tf.reshape(inputs, (-1, 7 * 7 * 64))  # flatten inputs
    inputs = tf.layers.dense(inputs, 1024, tf.nn.relu)

    # logits
    return tf.layers.dense(inputs, 10)

def convolution_nn(inputs, prefix='default'):

    features = 784
    conv1_filters = 32
    conv2_filters = 64
    mlp_units = 1024
    classes = 10

    # convolution
    with tf.name_scope(prefix + '_conv1'):
        sd = 1.0 / np.sqrt(float(features))
        kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=sd), name='kernel')
        conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.Variable(tf.constant(0.1, shape=[32]), name='bias')
        inputs = tf.nn.relu(tf.nn.bias_add(conv, bias), name='conv')
        inputs = tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')

    # convolution
    with tf.name_scope(prefix + '_conv2'):
        sd = 1.0 / np.sqrt(float(14 * 14 * 32))
        kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=sd), name='kernel')
        conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.Variable(tf.constant(0.1, shape=[64]), name='bias')
        inputs = tf.nn.relu(tf.nn.bias_add(conv, bias), name='conv')
        inputs = tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')

    # dense
    with tf.name_scope(prefix + '_inputs'):
        sd = 1.0 / np.sqrt(float(7 * 7 * 64))
        weights = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=sd), name='weights')
        bias = tf.Variable(tf.constant(0.1, shape=[1024]), name='bias')
        inputs = tf.reshape(inputs, (-1, 7 * 7 * 64))  # flatten inputs
        inputs =  tf.nn.relu(tf.matmul(inputs, weights) + bias)

    # logits
    with tf.name_scope(prefix + '_softmax'):
        sd = 1.0 / np.sqrt(float(1024))
        weights = tf.Variable(tf.truncated_normal([1024, 10], stddev=sd), name='weights')
        bias = tf.Variable(tf.constant(0.0, shape=[10]), name='bias')
        logits =  tf.matmul(inputs, weights) + bias

    return logits

def train(batch_size=128, max_batches=100000):
    """Perform simple training loop."""

    # load MNIST data
    images_train, labels_train, images_test, labels_test = load_data()
    train_size, train_shape = images_train.shape
    test_size, test_shape = images_test.shape

    # create placeholders
    inputs = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    labels = tf.placeholder(tf.int32, shape=(None,))

    # create network
    logits = convolution_nn(inputs)
    # logits = cnn(inputs)

    # define loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # define training op
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss)

    # define score
    predictions = tf.argmax(input=logits, axis=1)
    accuracy, accuracy_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

    # train...
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # required for tf.metrics.accuracy (uses local variables)

        batch = 0
        while batch < max_batches:

            idx = np.random.randint(0, train_size, batch_size)

            feed_dict = {
                inputs: np.reshape(images_train[idx, :], (-1, 28, 28, 1)),
                labels: labels_train[idx]
            }

            batch_loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)
            batch += 1

            if batch % 1000 == 0:

                idx = np.random.randint(0, test_size, batch_size * 10)  # *estimate* test score

                feed_dict = {
                    inputs: np.reshape(images_test[idx, :], (-1, 28, 28, 1)),
                    labels: labels_test[idx]
                }

                _ = sess.run(accuracy_op, feed_dict=feed_dict)  # updates total and count
                eval_acc = sess.run(accuracy, feed_dict=feed_dict)

                print("batch {:d}: loss = {:.4f}, acc = {:.4f}".format(batch, batch_loss, eval_acc))

            if batch % 10000 == 0:
                saver.save(sess, 'checkpoints/mnist-cnn', global_step=batch)

def test(batch_size=128):
    """Evaluate performance of trained model.

        Start by defining the same graph as the training routine, then restore the model `Variables` using `tf.Saver`.
    """

    # load MNIST data
    _, _, images_test, labels_test = load_data()
    test_size, test_shape = images_test.shape
    print("testing on {:d} examples...".format(test_size))

    # create placeholders
    inputs = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    labels = tf.placeholder(tf.int32, shape=(None,))

    # create network
    logits = convolution_nn(inputs)

    # define score
    predictions = tf.argmax(input=logits, axis=1)
    accuracy, accuracy_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

    # test...
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.local_variables_initializer())  # required for tf.metrics.accuracy (uses local variables)

        saver.restore(sess, 'checkpoints/mnist-cnn-100000')  # no extension!

        n = 0
        eval_acc = []
        while n < test_size:
            n += batch_size
            idx = np.random.randint(0, test_size, batch_size)
            feed_dict = {
                inputs: np.reshape(images_test[idx, :], (-1, 28, 28, 1)),
                labels: labels_test[idx]
            }
            _ = sess.run(accuracy_op, feed_dict=feed_dict)  # updates total and count
            eval_acc += sess.run([accuracy], feed_dict=feed_dict)

        print("eval accuracy = {:.4f}".format(np.mean(eval_acc)))

def visualize():
    """Visualize the first layer of activations."""

    # build the network
    inputs = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    logits = convolution_nn(inputs)

    # restore variables
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'checkpoints/mnist-cnn-100000')
        kernels = sess.run('default_conv1/kernel:0')

    # plot filters
    for i in range(kernels.shape[-1]):
        k = kernels[:, :, :, i]
        plt.subplot(4, 8, i + 1)
        plt.imshow(np.reshape(k, (5, 5)), cmap='Greys')
        plt.xticks([])
        plt.yticks([])
    plt.show()

if __name__ == "__main__":
    if FLAGS.mode == 'train':
        train()
    if FLAGS.mode == 'test':
        test()
    if FLAGS.mode == 'viz':
        visualize()
