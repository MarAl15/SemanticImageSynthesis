"""
    SPatially-Adaptive (DE)normalization (SPADE)

    Author: Mar Alguacil
"""
import tensorflow as tf


def spade(segmap, x, k_filters, nhidden=128, kernel_size=(3,3)):
    """
        Spatially-Adaptive Normalization.

          In the SPADE, the segmentation map (|segmap|) is first projected onto an embedding space and then convolved to
        produce the modulation parameters |alpha| and |beta|. The produced |alpha| and |beta| are multiplied and added to
        the normalized activation element-wise (Park et al.).
    """
    with tf.compat.v1.variable_scope('spade'):
        # Cast the semantic segmentation mask to float
        segmap = tf.cast(segmap, 'float')

        # Generate normalized activations
        normalized = batch_normalization(x)

        # Produce scaling and bias conditioned on semantic map
        segmap = tf.image.resize(images=segmap, size=tf.shape(x)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        actv = tf.keras.layers.Conv2D(nhidden, kernel_size, padding='same', activation='relu')(segmap)
        gamma = tf.keras.layers.Conv2D(k_filters, kernel_size, padding='same')(actv)
        beta = tf.keras.layers.Conv2D(k_filters, kernel_size, padding='same')(actv)


        return normalized * (1 + gamma) + beta



def batch_normalization(x, epsilon=1e-05):
    """"
        Batch normalization.
    """
    mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)

    # epsilon -> A small float number to avoid dividing by 0.
    return (x - mean) / tf.sqrt(variance + epsilon)
