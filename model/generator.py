"""
    Generator.

    Author: Mar Alguacil
"""
import tensorflow as tf
from model.spade import spade_resblk, leaky_relu


def generator(segmap, num_upsampling_layers='more', z=None, z_dim=256, nf=64):
    """
        Generator.

          The architecture of the generator consists mainly of a series of SPADE ResBlks with nearest
        neighbor upsampling.
    """
    num_up_layers = 6 if num_upsampling_layers=='more' else \
                    5 if num_upsampling_layers=='normal' else \
                    7
    if (num_up_layers==7 and num_upsampling_layers!='most'):
        raise ValueError('num_upsampling_layers [%s] not recognized' %
                          num_upsampling_layers)

    with tf.compat.v1.variable_scope('SPADE Generator'):
        # crop_size = segmap_width or segmapt_height
        s_dim = tf.shape(segmap)[1] // (2**num_up_layers)

        batch_size = tf.shape(segmap)[0]
        k_init = 16 * nf

        # Sample z from unit normal and reshape the tensor
        if z is None:
            z = tf.keras.backend.random_normal([batch_size, z_dim], dtype=tf.dtypes.float32)

        x = tf.keras.layers.Dense(k_init*s_dim*s_dim)(z)
        x = tf.reshape(x, [batch_size, s_dim, s_dim, k_init])

        # with tf.device("/cpu:0"):
        x = spade_resblk(segmap, x, k_init)

        x = upsample(x)
        x = spade_resblk(segmap, x, k_init)

        if num_upsampling_layers in ['more', 'most']:
            x = upsample(x)

        x = spade_resblk(segmap, x, k_init)
        x = upsample(x)
        x = spade_resblk(segmap, x, 8*nf)
        x = upsample(x)
        x = spade_resblk(segmap, x, 4*nf)
        x = upsample(x)
        x = spade_resblk(segmap, x, 2*nf)
        x = upsample(x)
        x = spade_resblk(segmap, x, nf)

        if num_upsampling_layers == 'most':
            x = upsample(x)
            x = spade_resblk(segmap, x, nf//2)

        x = leaky_relu(x)
        x = tf.keras.layers.Conv2D(3, kernel_size=3, padding='same')(x)

        return tf.keras.activations.tanh(x)

def upsample(x, scale_factor=2):
    _, h, w, _ = tf.shape(x)

    return tf.image.resize(images=x, size=[h * scale_factor, w * scale_factor],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)






