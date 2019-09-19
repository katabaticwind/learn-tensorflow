import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all messages
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
from time import time


max_steps = 10000
log_steps = 100
learning_rate = 1e-4
log_dir = './logs/run_{:.2f}'.format(time())


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
sample_size = 64
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

# 5. Create summary op
sample_images = tf.reshape(images_fake[:16, :], [-1, 28, 28, 1])
summary_op_d = tf.summary.merge(
    [
        tf.summary.scalar('loss_d', loss_d),
        tf.summary.histogram('prob_real', probs_real_real),
        tf.summary.histogram('prob_fake', probs_real_fake),
        tf.summary.histogram('logits_real', logits_real),
        tf.summary.histogram('logits_fake', logits_fake)
    ]
)
summary_op_g = tf.summary.merge(
    [
        tf.summary.scalar('loss_g', loss_g),
        tf.summary.image('images_fake', sample_images, max_outputs=16)
    ]
)

# III. Train the networks
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    loss = {'D': None, 'G': None}
    start = time()
    for step in range(max_steps):
        # 1. Sample z ~ N(0, 1)
        z = np.random.randn(sample_size, noise_size)

        # 2. Sample x
        idx = np.random.choice(range(len(images)), sample_size)
        x = images[idx, :]

        # 3. Update discriminator
        loss['D'], _, summary_d = sess.run(
            [loss_d, train_op_d, summary_op_d],
            feed_dict={images_pl: x, noise_pl: z}
        )

        # 4. Update generator
        loss['G'], _, summary_g = sess.run(
            [loss_g, train_op_g, summary_op_g],
            feed_dict={noise_pl: z}
        )

        # 5. Logging
        if step % log_steps == 0 and step > 0:
            writer.add_summary(summary_d, step)
            writer.add_summary(summary_g, step)
            print("step={:d}, loss_d={:.2f}, loss_g={:.2f}, elapsed={:.2f}".format(
                step,
                loss['D'],
                loss['G'],
                time() - start))
