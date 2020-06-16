from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d
import os,sys
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import io
import tensorflow.contrib.rnn as rnn


def laplace_attention(q, k, v, normalise=True):
    """Computes laplace exponential attention.

    Args:
      q: queries. tensor of shape [B,m,d_k].
      k: keys. tensor of shape [B,n,d_k].
      v: values. tensor of shape [B,n,d_v].
      scale: float that scales the L1 distance.
      normalise: Boolean that determines whether weights sum to 1.

    Returns:
      tensor of shape [B,m,d_v].
    """
    if len(q.get_shape().as_list()) != 3:
        q = tf.expand_dims(q, axis=0)
        k = tf.expand_dims(k, axis=0)
        v = tf.expand_dims(v, axis=0)
    d_k = tf.shape(q)[-1]
    scale = tf.sqrt(tf.cast(d_k, tf.float32))
    k = tf.expand_dims(k, axis=1)  # [B,1,n,d_k]
    q = tf.expand_dims(q, axis=2)  # [B,m,1,d_k]
    unnorm_weights = - tf.abs((k - q) / scale)  # [B,m,n,d_k]
    unnorm_weights = tf.reduce_sum(unnorm_weights, axis=-1)  # [B,m,n]
    if normalise:
        weight_fn = tf.nn.softmax
    else:
        weight_fn = lambda x: 1 + tf.tanh(x)
    weights = weight_fn(unnorm_weights)  # [B,m,n]
    rep = tf.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
    rep = tf.squeeze(rep, axis=0)
    return rep


def dot_product_attention(q, k, v, normalise):
    """Computes dot product attention.

    Args:
      q: queries. tensor of  shape [B,m,d_k].
      k: keys. tensor of shape [B,n,d_k].
      v: values. tensor of shape [B,n,d_v].
      normalise: Boolean that determines whether weights sum to 1.

    Returns:
      tensor of shape [B,m,d_v].
    """
    if len(q.get_shape().as_list()) != 3:
        q = tf.expand_dims(q, axis=0)
        k = tf.expand_dims(k, axis=0)
        v = tf.expand_dims(v, axis=0)
    d_k = tf.shape(q)[-1]
    scale = tf.sqrt(tf.cast(d_k, tf.float32))
    unnorm_weights = tf.einsum('bjk,bik->bij', k, q) / scale  # [B,m,n]
    if normalise:
        weight_fn = tf.nn.softmax
    else:
        weight_fn = tf.sigmoid
    weights = weight_fn(unnorm_weights)  # [B,m,n]
    rep = tf.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
    rep = tf.squeeze(rep, axis=0)
    return rep


def multihead_attention(q, k, v, num_heads=8):
    """Computes multi-head attention.

    Args:
      q: queries. tensor of  shape [B,m,d_k].
      k: keys. tensor of shape [B,n,d_k].
      v: values. tensor of shape [B,n,d_v].
      num_heads: number of heads. Should divide d_v.

    Returns:
      tensor of shape [B,m,d_v].
    """
    q = tf.expand_dims(q, axis=0)
    k = tf.expand_dims(k, axis=0)
    v = tf.expand_dims(v, axis=0)
    d_k = q.get_shape().as_list()[-1]
    d_v = v.get_shape().as_list()[-1]
    head_size = d_v / num_heads
    key_initializer = tf.random_normal_initializer(stddev=d_k ** -0.5)
    value_initializer = tf.random_normal_initializer(stddev=d_v ** -0.5)
    rep = tf.constant(0.0)
    for h in range(num_heads):
        o = dot_product_attention(
            tf.layers.Conv1D(head_size, 1, kernel_initializer=key_initializer,
                             name='wq%d' % h, use_bias=False, padding='VALID')(q),
            tf.layers.Conv1D(head_size, 1, kernel_initializer=key_initializer,
                             name='wk%d' % h, use_bias=False, padding='VALID')(k),
            tf.layers.Conv1D(head_size, 1, kernel_initializer=key_initializer,
                             name='wv%d' % h, use_bias=False, padding='VALID')(v),
            normalise=True)
        o = tf.expand_dims(o, axis=0)
        rep += tf.layers.Conv1D(d_v, 1, kernel_initializer=value_initializer,
                                name='wo%d' % h, use_bias=False, padding='VALID')(o)
    rep = tf.squeeze(rep, axis=0)
    return rep




"""
LSTM
"""
class LSTM:
    def __init__(self, name, layer_sizes, batch_size):
        """
        Initializes a multi layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer e.g. [100, 100, 100] returns a 3 layer, 100
                                                                                                        neuron bid-LSTM
        :param batch_size: The experiments batch size
        """
        self.reuse = False
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.name = name

    def __call__(self, inputs, initial_state, training=False):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        :param inputs: The inputs should be a list of shape [sequence_length, batch_size, 64]
        :param name: Name to give to the tensorflow op
        :param training: Flag that indicates if this is a training or evaluation stage
        :return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope("encoder"):

                lstm_cell = rnn.BasicLSTMCell(self.layer_sizes, forget_bias=1.0)
                # Get lstm cell output
                outputs, states = rnn.static_rnn(lstm_cell, inputs, initial_state, dtype=tf.float32)

            print("g out shape", tf.stack(outputs, axis=1).get_shape().as_list())

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return outputs, states


"""
biLSTM
"""

def bidirectionalLSTM(name, inputs, layer_sizes, initial_state_fw, initial_state_bw, time_major=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=layer_sizes)
        bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=layer_sizes)

        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            inputs,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            dtype=tf.float32,
            time_major=time_major)

    return tf.concat(outputs, axis=-1), output_states[0], output_states[1] #output_state_fw, output_state_bw


"""
Distances
"""
def KL_divergence(mu1, log_std1, mu2, log_std2):
    return tf.reduce_sum(
        2*(log_std2 - log_std1)
        + (tf.pow(tf.exp(log_std1), 4) + tf.square(mu1-mu2))/(2*tf.pow(tf.exp(log_std2), 4)) - 0.5)


"""
init Functions
"""
def init(name, shape, init_v, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=init_v, trainable=trainable)


"""
Kernel Functions
"""

def rand_features(bases, features, bias):
    # tf.random_normal()
    return tf.sqrt(2/bias.shape[0]) * tf.cos(tf.matmul(bases, features) + bias)

def dotp_kernel(feature1, feature2):
    return tf.matmul(feature1, feature2)

def cosine_dist(a, b):
    # sqrt(<a, b>) / (sqrt(<a, a>), sqrt(<b, b>))
    a = tf.reshape(a, [-1])
    b = tf.reshape(b, [-1])
    normalize_a = tf.nn.l2_normalize(a)
    normalize_b = tf.nn.l2_normalize(b)
    return tf.sqrt(tf.reduce_sum(tf.multiply(normalize_a, normalize_b)))


def normalize(x):
    max = tf.reduce_max(x)
    min = tf.reduce_min(x)

    res = max - min

    return (x-min)/res
    
"""
Probability Functions
"""


def sample_normal(mu, log_variance, num_samples, eps_=None):
    """
    Generate samples from a parameterized normal distribution.
    :param mu: tf tensor - mean parameter of the distribution.
    :param log_variance: tf tensor - log variance of the distribution.
    :param num_samples: np scalar - number of samples to generate.
    :return: tf tensor - samples from distribution of size num_samples x dim(mu).
    """
    if eps_ is not None:
        return mu + eps_ * tf.sqrt(tf.exp(log_variance))
    else:
        shape = tf.concat([tf.constant([num_samples]), tf.shape(mu)], axis=-1)
        eps = tf.random_normal(shape, dtype=tf.float32)
        return mu + eps * tf.sqrt(tf.exp(log_variance))


def multinoulli_log_density(inputs, logits):
    """
    Compute the log density under a multinoulli distribution.
    :param inputs: tf tensor - inputs with axis -1 as random vectors.
    :param logits: tf tensor - logits parameterizing Bernoulli distribution.
    :return: tf tensor - log density under Multinoulli distribution.
    """
    return -tf.nn.softmax_cross_entropy_with_logits(labels=inputs, logits=logits)


def gaussian_log_density(inputs, mu, logVariance):
    """
    Compute the log density under a parameterized normal distribution
    :param inputs: tf tensor - inputs with axis -1 as random vectors
    :param mu: tf tensor - mean parameter for normal distribution
    :param logVariance: tf tensor - log(sigma^2) of distribution
    :return: tf tensor - log density under a normal distribution
    """
    d = tf.cast(tf.shape(inputs)[-1], tf.float32)
    xc = inputs - mu
    return -0.5*(tf.reduce_sum((xc * xc) / tf.exp(logVariance), axis=-1)
                 + tf.reduce_sum(logVariance, axis=-1) + d * tf.log(2.0*np.pi))


"""
TensorFlow Network Support Functions
"""


def conv2d_pool_block(inputs, use_batch_norm, dropout_keep_prob, pool_padding, name):
    """
    A macro function that implements the following in sequence:
    - conv2d
    - batch_norm
    - relu activation
    - dropout
    - max_pool
    :param inputs: batch of feature maps.
    :param use_batch_norm: whether to use batch normalization or not.
    :param dropout_keep_prob: keep probability parameter for dropout.
    :param pool_padding: type of padding to use on the pooling operation.
    :param name: first part of the name used to scope this sequence of operations.
    :return: the processed batch of feature maps.
    """
    h = tf.layers.conv2d(
        inputs=inputs,
        strides=(1, 1),
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        kernel_initializer=xavier_initializer_conv2d(uniform=False),
        use_bias=False,
        name=(name + '_conv2d'),
        reuse=tf.AUTO_REUSE)

    if use_batch_norm:
        h = tf.contrib.layers.batch_norm(
            inputs=h,
            epsilon=1e-5,
            scope=(name + '_batch_norm'),
            reuse=tf.AUTO_REUSE)

    h = tf.nn.relu(features=h, name=(name + '_batch_relu'))

    h = tf.nn.dropout(x=h, keep_prob=dropout_keep_prob, name=(name + '_dropout'))

    h = tf.layers.max_pooling2d(inputs=h, pool_size=[2, 2], strides=2, padding=pool_padding, name=(name + '_pool'))

    return h


def dense_block(inputs, output_size, use_batch_norm, dropout_keep_prob, name):
    """
    A macro function that implements the following in sequence:
    - dense layer
    - batch_norm
    - relu activation
    - dropout
    :param inputs: batch of inputs.
    :param output_size: dimensionality of the output.
    :param use_batch_norm: whether to use batch normalization or not.
    :param dropout_keep_prob: keep probability parameter for dropout.
    :param name: first part of the name used to scope this sequence of operations.
    :return: batch of outputs.
    """
    h = tf.layers.dense(
        inputs=inputs,
        units=output_size,
        kernel_initializer=xavier_initializer(uniform=False),
        use_bias=False,
        name=(name + '_dense'),
        reuse=tf.AUTO_REUSE)

    if use_batch_norm:
        h = tf.contrib.layers.batch_norm(
            inputs=h,
            epsilon=1e-5,
            scope=(name + '_batch_norm'),
            reuse=tf.AUTO_REUSE)

    h = tf.nn.relu(features=h, name=(name + '_batch_relu'))

    h = tf.nn.dropout(x=h, keep_prob=dropout_keep_prob, name=(name + '_dropout'))

    return h


def dense_layer(inputs, output_size, activation, use_bias, name):
    """
    A simple dense layer.
    :param inputs: batch of inputs.
    :param output_size: dimensionality of the output.
    :param activation: activation function to use.
    :param use_bias: whether to have bias weights or not.
    :param name: name used to scope this operation.
    :return: batch of outputs.
     """
    return tf.layers.dense(
        inputs=inputs,
        units=output_size,
        kernel_initializer=xavier_initializer(uniform=False),
        use_bias=use_bias,
        bias_initializer=tf.random_normal_initializer(stddev=1e-3),
        activation=activation,
        name=name,
        reuse=tf.AUTO_REUSE)


def conv2d_transpose_layer(inputs, filters, activation, name):
    """
    A simple de-convolution layer.
    :param inputs: batch of inputs.
    :param filters: number of output filters.
    :param activation: activation function to use.
    :param name: name used to scope this operation.
    :return: batch of outputs.
     """
    return tf.layers.conv2d_transpose(
        inputs=inputs,
        filters=filters,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        activation=activation,
        data_format='channels_last',
        use_bias=False,
        kernel_initializer=xavier_initializer_conv2d(uniform=False),
        name=name,
        reuse=tf.AUTO_REUSE)



"""
print_and_log: Helper function to print to the screen and the log file.
"""


def print_and_log(log_file, message):
    print(message)
    log_file.write(message + '\n')


"""
get_log_files: Function that takes a path to a checkpoint directory and returns
a reference to a logfile and paths to the fully trained model and the model
with the best validation score.
"""


# def get_log_files(checkpoint_dir):
#     unique_dir_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#     unique_checkpoint_dir = os.path.join(checkpoint_dir, unique_dir_name)
#     if not os.path.exists(unique_checkpoint_dir):
#         os.makedirs(unique_checkpoint_dir)
#     checkpoint_path_validation = os.path.join(unique_checkpoint_dir, 'best_validation')
#     checkpoint_path_final = os.path.join(unique_checkpoint_dir, 'fully_trained')
#     logfile_path = os.path.join(unique_checkpoint_dir, 'log')
#     logfile = open(logfile_path, "w")
#     return logfile, checkpoint_path_validation, checkpoint_path_final

def get_log_files(checkpoint_dir, mode, shot):

    unique_checkpoint_dir = os.path.join(checkpoint_dir, str(shot)+'_checkpoint')
    if not os.path.exists(unique_checkpoint_dir):
        os.makedirs(unique_checkpoint_dir)
    checkpoint_path_validation = os.path.join(unique_checkpoint_dir, 'best_validation')
    checkpoint_path_final = os.path.join(unique_checkpoint_dir, 'fully_trained')
    logfile_path = os.path.join(unique_checkpoint_dir, mode + '_log')
    logfile = open(logfile_path, "w")
    return logfile, checkpoint_path_validation, checkpoint_path_final
"""
plot_image_strips: Function to plot view reconstruction image strips.
"""





def data_aug(images, crop_ratio, flip=False):
    shapes = images.get_shape().as_list()
    crop_size = int(crop_ratio * shapes[-2])
    shape_size = len(shapes)
    if shape_size == 4 or shape_size == 3:
        # flip
        if flip:
            images = tf.image.random_flip_left_right(images)
            images = tf.image.random_flip_up_down(images)

        # crop
        images = tf.image.random_crop(images, [tf.shape(images)[0], crop_size, crop_size, 3])
        images = tf.image.resize_images(images, size=[shapes[-3], shapes[-2]])

    else:
        sys.exit("Message to print to stderr")

    return images
