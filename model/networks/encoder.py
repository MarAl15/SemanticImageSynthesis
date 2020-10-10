"""
    Image Encoder.
"""
import tensorflow as tf
from model.networks.normalizations import instance_normalization
from utils.utils import Conv2d, weight_initializer, leaky_relu

def encoder(img_shape, crop_size, num_filters=16):
    """Image Encoder.

          The image enconder consists of 6 stride-2 convolutional layers followed by two linear layers
        to produce the mean and variance of the output distribution.
    """
    # kw = 3
    # pw = 1 #int(np.ceil((kw - 1.0) / 2))
    last_ndf = num_filters*8

    img = tf.keras.layers.Input(shape=img_shape[1:], batch_size=img_shape[0])
    x = img

    with tf.compat.v1.variable_scope('ImageEncoder'):
        if img_shape[1] != 256 or img_shape[2] != 256:
            x = tf.image.resize(images=x, size=[256, 256],
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

        x = tf.reshape(x, [img_shape[0], -1])

        mean = tf.keras.layers.Dense(256, kernel_initializer=weight_initializer())(x)
        variance = tf.keras.layers.Dense(256, kernel_initializer=weight_initializer())(x)

        return tf.keras.Model(inputs=img, outputs=[mean, variance])

