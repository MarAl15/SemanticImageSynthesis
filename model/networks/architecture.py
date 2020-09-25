"""
    SPADE Architecture.
"""
import tensorflow as tf
from utils.utils import Conv2d, leaky_relu
from model.networks.normalizations import spade, spectral_normalization


def spade_resblk(segmap, x, k_out, spectral_norm=False):
    """
        Residual block.

          Takes in the segmentation map (|segmap|) as input, learns the skip connection if necessary, and applies
        normalization first and then convolution.
    """
    k_in = tf.shape(x)[-1]
    k_middle = min(k_in, k_out)

    with tf.compat.v1.variable_scope('SPADE ResBlk'):
        dx = spade(segmap, x, k_in)
        dx = leaky_relu(dx)
        dx = Conv2d(dx, k_middle, kernel_size=3, padding=1)
        if spectral_norm:
            dx = spectral_normalization(dx)

        dx = spade(segmap, dx, k_middle)
        dx = leaky_relu(dx)
        dx = Conv2d(dx, k_out, kernel_size=3, padding=1)
        if spectral_norm:
            dx = spectral_normalization(dx)

        if k_in != k_out:
            x_s = spade(segmap, x, k_in)
            # x_s = leaky_relu(x_s)
            x_s = Conv2d(x_s, k_out, kernel_size=1, use_bias=False)
            if spectral_norm:
                x_s = spectral_normalization(x_s)
        else:
            x_s = x

        return x_s + dx
