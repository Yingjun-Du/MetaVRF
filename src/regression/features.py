import tensorflow as tf
from utilities import conv2d_pool_block, conv2d_transpose_layer, dense_layer, dense_block


def sin_function(x, d_theta):
    hidden_0 = tf.layers.dense(x, d_theta, tf.nn.relu, name='hidden_0',
        reuse=tf.AUTO_REUSE)
    hidden_1 = tf.layers.dense(hidden_0, d_theta, tf.nn.relu, name='hidden_1',
        reuse=tf.AUTO_REUSE)
    return hidden_1


