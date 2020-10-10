"""
    Loss functions.
"""
import tensorflow as tf
from model.networks.architecture import VGG19, initialize_vgg19


def hinge_loss_discriminator(real, fake):
    """Computes hinge loss for discriminator."""
    # Real and fake loss
    return -tf.reduce_mean([tf.reduce_mean(tf.minimum(real_discr[-1] - 1, 0.0)) for real_discr in real]), \
           -tf.reduce_mean([tf.reduce_mean(tf.minimum(-fake_discr[-1] - 1, 0.0)) for fake_discr in fake])


def hinge_loss_generator(fake):
    """Computes hinge loss for generator."""
    return -tf.reduce_mean([tf.reduce_mean(fake_discr[-1]) for fake_discr in fake])


def kld_loss(mean, log_var):
    """Computes KL divergence loss."""
    return 0.5 * tf.math.reduce_sum(mean*mean + tf.math.exp(log_var) - 1 - log_var)


def l1_loss(x, y):
    """Computes L1 loss."""
    return tf.math.reduce_mean(tf.math.abs(x - y))


def vgg_loss(x, y, trainable=False):
    """Computes VGG loss."""
    x = (x + 1) * 127.5
    y = (y + 1) * 127.5
    slices = initialize_vgg19(trainable)
    x_vgg = VGG19(tf.keras.applications.vgg19.preprocess_input(x), trainable, slices)
    y_vgg = VGG19(tf.keras.applications.vgg19.preprocess_input(y), trainable, slices)

    loss = 0.03125 * l1_loss(x_vgg[0], tf.stop_gradient(y_vgg[0]))
    loss += 0.0625 * l1_loss(x_vgg[1], tf.stop_gradient(y_vgg[1]))
    loss += 0.125 * l1_loss(x_vgg[2], tf.stop_gradient(y_vgg[2]))
    loss += 0.25 * l1_loss(x_vgg[3], tf.stop_gradient(y_vgg[3]))
    loss += l1_loss(x_vgg[4], tf.stop_gradient(y_vgg[4]))

    return loss
