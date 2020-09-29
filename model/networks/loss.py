"""
    Loss functions.
"""
import tensorflow as tf


def hinge_loss_discriminator(real, fake):
    """
        Computes hinge loss for discriminator.
    """
    # Real and fake loss
    return -tf.reduce_mean(tf.minimum(real - 1, 0.0)), -tf.reduce_mean(tf.minimum(-fake - 1, 0.0))

def hinge_loss_generator(fake):
    """
        Computes hinge loss for generator.
    """
    return -tf.reduce_mean(fake)


def kld_loss(mean, log_var):
    """
        Computes KL divergence loss.
    """
    return 0.5 * tf.math.reduce_sum(mean*mean + tf.math.exp(log_var) - 1 - log_var)
