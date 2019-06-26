import tensorflow as tf

def dense(inputs, hidden_units, activation, dtype = tf.float32):
    """Create a fully-connected layer.

        # Arguments
        - `inputs::Tensor`: assumed to have shape (None, inputs_units).
        - `hidden_units::Int`: number of hidden units in layer.
        - `activation`: activation to apply following inner product.

        # Returns
        - `output (tf.Tensor)`
    """
    inputs_units = tf.shape(inputs)[1]
    weights = tf.Variable(name='weights', shape=(input_units, hidden_units), dtype = dtype)
    bias = tf.Variable(name='bias', shape=(hidden_units, ), dtype=dtype)
    return activation(tf.add(bias, tf.matmul(inputs, weights)))

def conv2d(inputs):
    pass
