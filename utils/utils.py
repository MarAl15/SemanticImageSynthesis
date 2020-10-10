"""
   Auxiliar functions.
"""

import tensorflow as tf
import tensorflow_addons as tfa


############
#  LAYERS  #
############
def Conv2d(x, filters, kernel_size, padding=0, mode='CONSTANT', strides=1, spectral_norm=False, use_bias=True):
    """Applies a 2D convolution by padding the tensor first if required."""
    # padding controls the amount of implicit zero-paddings on both sides for padding number of points for each dimension.
    if padding > 0:
        x = tf.pad(x, tf.constant([[0,0], [padding, padding], [padding, padding], [0,0]]), mode)

    if spectral_norm:
        return tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, use_bias=use_bias), power_iterations=2)(x)
    return tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, use_bias=use_bias)(x)


##########################
#  ACTIVATION FUNCTIONS  #
##########################
def leaky_relu(x):
    """Computes the Leaky ReLU activation function."""
    return tf.nn.leaky_relu(x, 0.2)
