"""
    Model.
"""
import tensorflow as tf
from model.networks.loss import *


class Model(object):
    def __init__(self, args, generator, discriminator, encoder=None, training=False):
        # To use or not image encoder
        self.use_vae = args.use_vae
        if self.use_vae:
            # Encoder
            self.encoder = encoder
        # Generator
        self.generator = generator
        # Discriminator
        self.discriminator = discriminator

        # Loss weights
        self.lambda_kld = args.lambda_kld
        self.lambda_features = args.lambda_features
        self.lambda_vgg = args.lambda_vgg
        # To use or not discriminator feature matching loss
        self.no_feature_loss = args.no_feature_loss
        # To use or not VGG loss
        self.no_vgg_loss = args.no_vgg_loss

        # Training?
        self.training = training

    def create_optimizers(self, generator_loss, discriminator_loss, lr, beta1, beta2, no_TTUR=True):
        """Constructs the generator and discriminator optimizers."""
        train_vars = tf.compat.v1.trainable_variables()
        generator_vars = [var for var in train_vars if 'Encoder' in var.name or 'Generator' in var.name]
        discriminator_vars = [var for var in train_vars if 'Discriminator' in var.name]

        (g_lr, d_lr) = (lr/2, lr*2) if not no_TTUR else (lr, lr)

        # generator_optimizer, discriminator_optimizer
        return tf.compat.v1.train.AdamOptimizer(g_lr, beta1=beta1, beta2=beta2).\
                  minimize(generator_loss, var_list=generator_vars), \
               tf.compat.v1.train.AdamOptimizer(d_lr, beta1=beta1, beta2=beta2).\
                  minimize(discriminator_loss, var_list=discriminator_vars),


    def compute_losses(self, segmap_image, real_image):
        """Calculates the different losses related to the generator and discriminator.

            Generator:
              - Generator hinge loss for the fake image.
              - If use_vae is specified, KLD loss.
              - If no_feature_loss is not specified, average L1 loss of each intermediate output.
              - If no_vgg_loss is not specified, VGG loss.
            Discriminator:
              - Discriminator hinge loss for the real and fake image.
        """
        generator_losses, discriminator_losses = {}, {}

        if self.use_vae:
            fake_image, generator_losses['KLD'] = self.generate_fake(segmap_image)
        else:
            fake_image, _ = self.generate_fake(segmap_image, real_image)

        pred_real, pred_fake = self.discriminate(segmap_image, real_image, fake_image,
                                                 get_intermediate_features=(not self.no_feature_loss))

        generator_losses['Hinge'] = hinge_loss_generator(pred_fake)

        if not self.no_feature_loss:
            generator_losses['Feature'] = 0.0
            # self.num_discriminators = len(pred_fake)
            for i in range(self.num_discriminators):
                # Last output is the final prediction, so it is excluded
                for j in range(len(pred_fake[i])-1): # for each layer output
                    # generator_losses['Feature'] += l1_loss(pred_fake[i][j], tf.stop_gradient(pred_real[i][j]))
                    generator_losses['Feature'] += l1_loss(pred_fake[i][j], pred_real[i][j])
            generator_losses['Feature'] *= self.lambda_features/self.num_discriminators

        if not self.no_vgg_loss:
            generator_losses['VGG'] = vgg_loss(real_image, fake_image, self.training) * self.lambda_vgg

        discriminator_losses['HingeReal'], discriminator_losses['HingeFake'] = hinge_loss_discriminator(pred_real, pred_fake)

        return generator_losses, discriminator_losses, fake_image


    def discriminate(self, segmap_image, fake_image, real_image, get_intermediate_features=False):
        """Returns the probability of the real_image and fake_image to be real."""
        #pred_real, pred_fake
        return discriminator(segmap_image, real_image, self.num_discriminators, self.num_discriminator_filters, self.num_discriminator_layers,
                             get_intermediate_features=get_intermediate_features), \
               discriminator(segmap_image, fake_image, self.num_discriminators, self.num_discriminator_filters, self.num_discriminator_layers,
                             get_intermediate_features=get_intermediate_features, reuse=tf.compat.v1.AUTO_REUSE)


    def generate_fake(self, segmap_image, real_image=None):
        """Creates a fake image from the segmentation map of another image."""
        if self.use_vae:
            mean, log_var = self.encoder(real_image)
            z = tf.math.multiply(tf.random.normal(mean.shape), tf.math.exp(0.5 * log_var)) + mean
            KLD_loss = kld_loss(mean, log_var) * self.lambda_kld

            return self.generator([segmap_image, z]), KLD_loss

        return self.generator(segmap_image), None


# def use_gpu():
    # return len(tf.config.list_physical_devices('GPU'))>0
