"""
    Loss functions.
"""
import tensorflow as tf


def hinge_loss_discriminator(real, fake):
    """
        Computes hinge loss for discriminator.
    """
    # Real and fake loss
    return -tf.reduce_mean([tf.reduce_mean(tf.minimum(real_discr[-1] - 1, 0.0)) for real_discr in real]), \
           -tf.reduce_mean([tf.reduce_mean(tf.minimum(-fake_discr[-1] - 1, 0.0)) for fake_discr in fake])


def hinge_loss_generator(fake):
    """
        Computes hinge loss for generator.
    """
    return -tf.reduce_mean([tf.reduce_mean(fake_discr[-1]) for fake_discr in fake])


def kld_loss(mean, log_var):
    """
        Computes KL divergence loss.
    """
    return 0.5 * tf.math.reduce_sum(mean*mean + tf.math.exp(log_var) - 1 - log_var)


def l1_loss(x, y):
    """
        Computes L1 loss.
    """
    return tf.math.reduce_mean(tf.math.abs(x, y))
