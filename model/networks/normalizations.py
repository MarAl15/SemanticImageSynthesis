"""
    Normalizations.
"""
import tensorflow as tf
import tensorflow_addons as tfa
from utils.utils import Conv2d


def spade(segmap, x, k_filters, nhidden=128, kernel_size=3):
    """Applies SPatially-Adaptive (DE)normalization (SPADE).

          In the SPADE, the segmentation map (|segmap|) is first projected onto an embedding space and then convolved to
        produce the modulation parameters |alpha| and |beta|. The produced |alpha| and |beta| are multiplied and added to
        the normalized activation element-wise (Park et al.).
    """
    with tf.compat.v1.variable_scope('SPADE'):
        # Generate normalized activations
        normalized = batch_normalization(x)

        # Produce scaling and bias conditioned on semantic map
        segmap = tf.image.resize(images=segmap, size=tf.shape(x)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        pw = kernel_size//2
        actv = Conv2d(segmap, nhidden, kernel_size, padding=pw)
        actv = tf.nn.relu(actv)

        gamma = Conv2d(actv, k_filters, kernel_size, padding=pw)
        beta = Conv2d(actv, k_filters, kernel_size, padding=pw)


        return normalized * (1 + gamma) + beta


def batch_normalization(x, epsilon=1e-05):
    """"Applies Batch Normalization."""
    mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)

    # epsilon -> A small float number to avoid dividing by 0.
    return (x - mean) / tf.sqrt(variance + epsilon)


def instance_normalization(x, epsilon=1e-05):
    """"Applies Instance Normalization"""
    return tfa.layers.InstanceNormalization(epsilon=epsilon, center=True, scale=True)(x)
