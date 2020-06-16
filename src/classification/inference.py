import tensorflow as tf
from utilities import dense_layer, sample_normal


def inference_block(inputs, d_theta, output_units, name):
    """
    Three dense layers in sequence.
    :param inputs: batch of inputs.
    :param d_theta: dimensionality of the intermediate hidden layers.
    :param output_units: dimensionality of the output.
    :param name: name used to scope this operation.
    :return: batch of outputs.
     """
    h = dense_layer(inputs, d_theta, tf.nn.elu, True, name + '1')
    h = dense_layer(h, d_theta, tf.nn.elu, True, name + '2')
    h = dense_layer(h, output_units, None, True, name + '3')
    return h

