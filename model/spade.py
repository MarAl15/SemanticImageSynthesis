"""
    SPADE Architecture.

    Author: Mar Alguacil
"""
import tensorflow as tf


def spade_resblk(segmap, x, k_out):
    """
        Residual block.

        Takes in the segmentation map (|segmap|) as input, learns the skip connection if necessary, and applies normalization
        first and then convolution.
    """
    k_in = tf.shape(x)[-1]
    k_middle = min(k_in, k_out)

    with tf.compat.v1.variable_scope('SPADE ResBlk'):
        dx = spade(segmap, x, k_in)
        dx = leaky_relu(dx)
        dx = tf.keras.layers.Conv2D(k_middle, kernel_size=3, padding='same')(dx)

        dx = spade(segmap, dx, k_middle)
        dx = leaky_relu(dx)
        dx = tf.keras.layers.Conv2D(k_out, kernel_size=3, padding='same')(dx)

        if (k_in!=k_out):
            x_s = spade(segmap, x, k_in)
            # x_s = leaky_relu(x_s)
            x_s = tf.keras.layers.Conv2D(k_out, kernel_size=1, padding='same', use_bias=False)(x_s)
        else:
            x_s = x

        return x_s + dx

def leaky_relu(x):
    """
        Compute the Leaky ReLU activation function.
    """
    return tf.nn.leaky_relu(x, 0.2)

def spade(segmap, x, k_filters, nhidden=128, kernel_size=(3,3)):
    """
        SPatially-Adaptive (DE)normalization (SPADE)

          In the SPADE, the segmentation map (|segmap|) is first projected onto an embedding space and then convolved to
        produce the modulation parameters |alpha| and |beta|. The produced |alpha| and |beta| are multiplied and added to
        the normalized activation element-wise (Park et al.).
    """
    with tf.compat.v1.variable_scope('SPADE'):
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
