import tensorflow as tf

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
