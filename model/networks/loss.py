"""
    Loss functions.
"""
import tensorflow as tf
from model.networks.architecture import VGG19


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



class VGGLoss(tf.keras.Model):
    """Computes VGG loss."""
    def __init__(self, shape):
        super(VGGLoss, self).__init__()
        self.vgg19 = VGG19(shape)

    def call(self, x, y):
        # Expects float input in [-1,1]
        # VGG19 works with values in the range [0, 255]
        x = (x + 1) * 127.5
        y = (y + 1) * 127.5
        x_vgg = self.vgg19(tf.keras.applications.vgg19.preprocess_input(x))
        y_vgg = self.vgg19(tf.keras.applications.vgg19.preprocess_input(y))

        loss = 0.03125 * l1_loss(x_vgg[0], tf.stop_gradient(y_vgg[0]))
        loss += 0.0625 * l1_loss(x_vgg[1], tf.stop_gradient(y_vgg[1]))
        loss += 0.125 * l1_loss(x_vgg[2], tf.stop_gradient(y_vgg[2]))
        loss += 0.25 * l1_loss(x_vgg[3], tf.stop_gradient(y_vgg[3]))
        loss += l1_loss(x_vgg[4], tf.stop_gradient(y_vgg[4]))

        return loss
