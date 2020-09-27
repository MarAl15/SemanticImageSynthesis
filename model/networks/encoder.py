"""
    Image Encoder.
"""
import tensorflow as tf
from model.networks.normalizations import instance_normalization
from utils.utils import Conv2d, leaky_relu

def encoder(x, crop_size, num_filters=64):
    """
        Image Encoder.

          The image enconder consists of 6 stride-2 convolutional layers followed by two linear layers
        to produce the mean and variance of the output distribution.
    """
    # kw = 3
    # pw = 1 #int(np.ceil((kw - 1.0) / 2))
    last_ndf = num_filters*8

    if tf.shape(x)[1] != 256 or tf.shape(x)[2] != 256:
        tf.image.resize(images=x, size=[256, 256],
                        method=tf.image.ResizeMethod.BILINEAR)

    x = Conv2d(x, num_filters, kernel_size=3, padding=1, strides=2)
    x = instance_normalization(x)
    x = leaky_relu(x)

    x = Conv2d(x, num_filters*2, kernel_size=3, padding=1, strides=2)
    x = instance_normalization(x)
    x = leaky_relu(x)

    x = Conv2d(x, num_filters*4, kernel_size=3, padding=1, strides=2)
    x = instance_normalization(x)
    x = leaky_relu(x)

    x = Conv2d(x, last_ndf, kernel_size=3, padding=1, strides=2)
    x = instance_normalization(x)
    x = leaky_relu(x)

    x = Conv2d(x, last_ndf, kernel_size=3, padding=1, strides=2)
    x = instance_normalization(x)

    if crop_size >= 256:
        x = leaky_relu(x)
        x = Conv2d(x, last_ndf, kernel_size=3, padding=1, strides=2)
        x = instance_normalization(x)

    x = leaky_relu(x)

    x = tf.reshape(x, [tf.shape(x)[0], -1])
    return tf.keras.layers.Dense(256)(x), tf.keras.layers.Dense(256)(x) # mean, variance

