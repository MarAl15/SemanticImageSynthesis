"""
    Discriminator.
"""
import tensorflow as tf
import tensorflow_addons as tfa
from model.networks.normalizations import spectral_normalization
from utils.utils import Conv2d, leaky_relu

def discriminator(segmap, x, num_discriminators=2, num_filters=64, num_layers=4, get_intermediate_features=True):
    """
        Discriminator.

          The architecture of the discriminator uses a multi-scale design with the Instance
        Normalization and applies the Spectral Normalization to almost all the convolutional
        layers of the discriminator.
    """
    # kw = 4
    # padw = 1 #int(np.ceil((kw - 1.0) / 2))

    with tf.compat.v1.variable_scope('Discriminator'):
        result = []

        for i in range(num_discriminators):
            concat = tf.concat([segmap, x], -1)

            nf = num_filters
            intermediate_outputs = []

            concat = Conv2d(concat, nf, kernel_size=4, padding=1, strides=2)
            concat = leaky_relu(concat)
            intermediate_outputs.append(concat)

            for n in range(1, num_layers):
                nf = min(nf*2, 512)
                stride = 1 if n == num_layers-1 else 2

                concat = Conv2d(concat, nf, kernel_size=4, padding=1, strides=stride)
                concat = spectral_normalization(concat)
                concat = tfa.layers.InstanceNormalization(epsilon=1e-05, center=True, scale=True)(concat)
                concat = leaky_relu(concat)
                intermediate_outputs.append(concat)

            concat = Conv2d(concat, 1, kernel_size=4, padding=1, strides=1)
            intermediate_outputs.append(concat)

            if i != num_discriminators-1:
                segmap = downsample(segmap)
                x = downsample(x)

            result.append(intermediate_outputs if get_intermediate_features else [concat])

        return result

def downsample(x):
    """
        Performs the average pooling.
    """
    return tf.keras.layers.AveragePooling2D(pool_size=3, strides=2, padding='SAME')(x)
