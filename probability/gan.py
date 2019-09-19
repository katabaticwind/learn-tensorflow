import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

max_steps = 20000
log_steps = 100
learning_rate = 1e-5


def load_data():
    print("Loading data...")
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    return mnist.train.images
    # images = np.split(images, images.shape[0], axis=0)
    # return [img.flatten() for img in images]


def mlp(inputs, sizes, activation=tf.nn.relu, scope='', reuse=None):
    """Create a multilayer perceptron model."""
    with tf.variable_scope(name_or_scope=scope, reuse=reuse):
        for size in sizes[:-1]:
            inputs = tf.layers.dense(inputs, size, activation)
        if sizes[-1] == 1:
            logits = tf.reshape(tf.layers.dense(inputs, sizes[-1]), [-1])
        else:
            logits = tf.layers.dense(inputs, sizes[-1])
    return logits


# I. Import MNIST data
images = load_data()
sample_size = 32
noise_size = 8
image_size = 28 * 28

# II. Setup the graph

# 1. Create placeholders
images_pl = tf.placeholder(tf.float32, [None, image_size])
noise_pl = tf.placeholder(tf.float32, [None, noise_size])

# 2. Create networks
images_fake = mlp(noise_pl, [64, 32, image_size], scope='generator')
logits_real = mlp(images_pl, [64, 32, 1], scope='discriminator')
logits_fake = mlp(images_fake, [64, 32, 1], scope='discriminator', reuse=True)

# 3. Create loss
probs_real_real = tf.exp(logits_real) / (1 + tf.exp(logits_real))
probs_fake_fake = 1 / (1 + tf.exp(logits_fake))
probs_real_fake = tf.exp(logits_fake) / (1 + tf.exp(logits_fake))
loss_d = -tf.reduce_mean(tf.log(probs_real_real) + tf.log(probs_fake_fake))
loss_g = -tf.reduce_mean(tf.log(probs_real_fake) - tf.log(probs_fake_fake))

# 4. Create training op
variables_d = tf.get_collection('trainable_variables', 'discriminator')
variables_g = tf.get_collection('trainable_variables', 'generator')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op_d = optimizer.minimize(loss_d, var_list=variables_d)
train_op_g = optimizer.minimize(loss_g, var_list=variables_g)


# III. Train the networks
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss = {'D': None, 'G': None}
    for iter in range(max_steps):
        # 1. Sample z ~ N(0, 1)
        z = np.random.randn(sample_size, noise_size)

        # 2. Sample x
        idx = np.random.choice(range(len(images)), sample_size)
        x = images[idx, :]

        # 3. Update discriminator
        feed_dict = {images_pl: x, noise_pl: z}
        loss['D'], _ = sess.run([loss_d, train_op_d], feed_dict=feed_dict)

        # 4. Update generator
        feed_dict = {noise_pl: z}
        loss['G'], _ = sess.run([loss_g, train_op_g], feed_dict=feed_dict)

        # 5. Logging
        if iter % log_steps == 0 and iter > 0:
            print("iter={}, loss_d={}, loss_g={}".format(
                iter,
                loss['D'],
                loss['G']))

    # IV. Generate samples

    # 1. Sample z ~ N(0, 1)
    z = np.random.randn(16, noise_size)

    # 2. Perform forward pass
    img = sess.run(images_fake, feed_dict={noise_pl: z})

    # 3. Display the result
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(img[i, :].reshape(28, 28))
