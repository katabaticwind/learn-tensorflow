import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'test', "Run script in training or testing mode.")

def load_data():
    print("Loading data...")
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    images_train = mnist.train.images
    labels_train = np.asarray(mnist.train.labels, dtype=np.int32)
    images_test = mnist.test.images
    labels_test = np.asarray(mnist.test.labels, dtype=np.int32)
    return images_train, labels_train, images_test, labels_test

def mlp(inputs, sizes, activation):
    """Create a multi-layer perceptron network."""
    inputs = tf.layers.dense(inputs, sizes[0], activation)
    for size in sizes[1:-1]:
        inputs = tf.layers.dense(inputs, size, activation)
    logits = tf.layers.dense(inputs, sizes[-1])
    return logits

def multilayer_perceptron(inputs, sizes, activation, scope=''):
    """
        Create a multi-layer perceptron network using **low-level API**.

        # Arguments
        * sizes (list): [num_inputs, nn1, nn2, ..., num_classes]
        * inputs (placeholder): [batch_size x num_inputs]
    """

    with tf.name_scope(scope):
        # hidden layers
        for i in range(1, len(sizes) - 1):
            sd = 1.0 / np.sqrt(float(sizes[i - 1]))
            weights = tf.Variable(tf.truncated_normal([sizes[i - 1], sizes[i]], stddev=sd), name='weights')
            bias = tf.Variable(tf.constant(0.1, shape=[sizes[i]]), name='bias')
            inputs =  activation(tf.matmul(inputs, weights) + bias)

        # logits
        sd = 1.0 / np.sqrt(float(sizes[-2]))
        weights = tf.Variable(tf.truncated_normal([sizes[-2], sizes[-1]], stddev=sd), name='weights')
        bias = tf.Variable(tf.constant(0.0, shape=[sizes[-1]]), name='bias')
        logits =  tf.matmul(inputs, weights) + bias
    return logits

def train(batch_size=128, max_batches=50000):
    """Perform simple training loop."""

    # load MNIST data
    images_train, labels_train, images_test, labels_test = load_data()
    train_size, train_shape = images_train.shape
    test_size, test_shape = images_test.shape

    # create placeholders
    inputs = tf.placeholder(tf.float32, shape=(None, train_shape))
    labels = tf.placeholder(tf.int32, shape=(None,))

    # create network
    logits = multilayer_perceptron(inputs, [784, 32, 16, 8, 10], tf.nn.relu)
    # logits = mlp(inputs, [32, 16, 8, 10], tf.nn.relu)

    # define loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # define training op
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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
                inputs: images_train[idx, :],
                labels: labels_train[idx]
            }

            batch_loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)
            batch += 1

            if batch % 1000 == 0:

                idx = np.random.randint(0, test_size, batch_size * 10)

                feed_dict = {
                    inputs: images_test[idx, :],
                    labels: labels_test[idx]
                }

                _ = sess.run(accuracy_op, feed_dict=feed_dict)  # updates total and count
                eval_acc = sess.run(accuracy, feed_dict=feed_dict)

                print("batch {:d}: loss = {:.4f}, acc = {:.4f}".format(batch, batch_loss, eval_acc))

            if batch % 10000 == 0:
                saver.save(sess, 'checkpoints/mnist-mlp', global_step=batch)

def test(batch_size=128):
    """Evaluate performance of trained model.

        Start by defining the same graph as the training routine, then restore the model `Variables` using `tf.Saver`.
    """

    # load MNIST data
    _, _, images_test, labels_test = load_data()
    test_size, test_shape = images_test.shape
    print("testing on {:d} examples...".format(test_size))

    # create placeholders
    inputs = tf.placeholder(tf.float32, shape=(None, 784))
    labels = tf.placeholder(tf.int32, shape=(None,))

    # create network
    logits = multilayer_perceptron(inputs, [784, 32, 16, 8, 10], tf.nn.relu)
    # logits = mlp(inputs, [32, 16, 8, 10], tf.nn.relu)

    # define score
    predictions = tf.argmax(input=logits, axis=1)
    accuracy, accuracy_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

    # test...
    saver = tf.train.Saver()
    with tf.Session() as sess:

        # sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # required for tf.metrics.accuracy (uses local variables)

        saver.restore(sess, 'checkpoints/mnist-mlp-50000')  # no extension!

        n = 0
        eval_acc = []
        while n < test_size:
            n += batch_size
            idx = np.random.randint(0, test_size, batch_size)
            feed_dict = {
                inputs: images_test[idx, :],
                labels: labels_test[idx]
            }
            _ = sess.run(accuracy_op, feed_dict=feed_dict)  # updates total and count
            eval_acc += sess.run([accuracy], feed_dict=feed_dict)

        print("eval accuracy = {:.4f}".format(np.mean(eval_acc)))

def visualize():
    """Visualize the first layer of activations."""

    # build the network
    inputs = tf.placeholder(tf.float32, shape=(None, 784))
    logits = multilayer_perceptron(inputs, [784, 32, 16, 8, 10], tf.nn.relu)

    # restore variables
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'checkpoints/mnist-mlp-50000')
        weights = sess.run('weights:0')

    # plot filters
    for i in range(weights.shape[1]):
        w = weights[:, i]
        plt.subplot(4, 8, i + 1)
        plt.imshow(np.reshape(w, (28, 28)))
        plt.xticks([])
        plt.yticks([])
    plt.show()

if __name__ == "__main__":
    if FLAGS.mode == 'train':
        train()
    if FLAGS.mode == 'test':
        test()



def clone(target, source, graph = tf.get_default_graph()):
    """Clone trainable variables from source to target.

        Assumes that `source` and `target` have identical variables (with same ordering).

        # Arguments
        - `target`: name scope of variables to clone to.
        - `source`: name scope of variables to clone from.

        # Returns
        - `clone_ops`: `Operation` that performs cloning.
    """
    with graph.as_default():
        source = graph.get_collection('trainable_variables', source)
        # print(source)
        target = graph.get_collection('trainable_variables', target)
        # print(target)
        clone_ops = [tf.assign(t, s, name='clone') for t,s in zip(target, source)]
        return clone_ops

def cloning_example():
    # Using default graph...
    inputs = tf.placeholder(tf.float32, shape=(None, 784))
    logits_default = multilayer_perceptron(inputs, [784, 32, 16, 10], tf.nn.relu, 'default')
    logits_clone = multilayer_perceptron(inputs, [784, 32, 16, 10], tf.nn.relu, 'clone')

    vars_default = tf.get_default_graph().get_collection('trainable_variables', 'default')
    print("DEFAULT NAME SCOPE:")
    for var in vars_default:
        print(var)

    vars_clone = tf.get_default_graph().get_collection('trainable_variables', 'clone')
    print("CLONE NAME SCOPE:")
    for var in vars_clone:
        print(var)

    clone_ops = clone('default', 'clone')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x = np.random.randn(1, 784)
        print(sess.run(logits_default, feed_dict={inputs: x}))
        print(sess.run(logits_clone, feed_dict={inputs: x}))
        print("Cloning default -> clone...")
        sess.run(clone_ops)
        print(sess.run(logits_default, feed_dict={inputs: x}))
        print(sess.run(logits_clone, feed_dict={inputs: x}))

def cloning_example_2():
    # Using specific graph...
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=(None, 784))
        logits_default = multilayer_perceptron(inputs, [784, 32, 16, 10], tf.nn.relu, 'default')
        logits_clone = multilayer_perceptron(inputs, [784, 32, 16, 10], tf.nn.relu, 'clone')

        vars_default = graph.get_collection('trainable_variables', 'default')
        print("DEFAULT NAME SCOPE:")
        for var in vars_default:
            print(var)

        vars_clone = graph.get_collection('trainable_variables', 'clone')
        print("CLONE NAME SCOPE:")
        for var in vars_clone:
            print(var)

    clone_ops = clone('default', 'clone', graph)

    with graph.as_default():
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        x = np.random.randn(1, 784)
        print(sess.run(logits_default, feed_dict={inputs: x}))
        print(sess.run(logits_clone, feed_dict={inputs: x}))
        print("Cloning default -> clone...")
        sess.run(clone_ops)
        print(sess.run(logits_default, feed_dict={inputs: x}))
        print(sess.run(logits_clone, feed_dict={inputs: x}))


"""
    Saving and Restoring

    # Save
    saver = tf.train.Saver()
    with tf.Session() as sess:

        # do some stuff...

        saver.save(sess, ckpt_dir)

    # Restore
    saver = tf.train.Saver()
    with tf.Session() as sess:

        saver.restore(sess, ckpt_dir)  # restore **variables** from checkpoint

        # do some stuff...
"""
