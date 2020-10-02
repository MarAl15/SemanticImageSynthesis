"""
    Model.
"""
import tensorflow as tf

from model.networks.encoder import encoder
from model.networks.generator import generator
from model.networks.discriminator import discriminator
from model.networks.loss import *


class Model(object):
    def __init__(self, sess, args):
        self.model_name = 'Model'

        self.sess = sess
        self.opt = args

    def compute_generator_loss(self, segmap_image, real_image):
        """Calculates the different losses related to the generator.

            Generator hinge loss for the fake image.
            If use_vae is specified, KLD loss.
            If no_feature_loss is not specified, average L1 loss of each intermediate output.
            If no_vgg_loss is not specified, VGG loss.
        """
        generator_losses = {}

        if self.opt.use_vae:
            fake_image, generator_losses['KLD'] = self.generate_fake(segmap_image, real_image,
                                                                     compute_kld_loss=True)
        else:
            fake_image, _ = self.generate_fake(segmap_image, real_image)

        pred_real, pred_fake = self.discriminate(segmap_image, real_image, fake_image,
                                                 get_intermediate_features=(not self.opt.no_feature_loss))

        generator_losses['Hinge'] = hinge_loss_generator(pred_fake)

        if not self.opt.no_feature_loss:
            generator_losses['Feature'] = 0.0
            # self.opt.num_discriminators = len(pred_fake)
            for i in range(self.opt.num_discriminators):
                # Last output is the final prediction, so it is excluded
                for j in range(len(pred_fake[i])-1): # for each layer output
                    # generator_losses['Feature'] += l1_loss(pred_fake[i][j], pred_real[i][j]) * self.opt.lambda_features/self.opt.num_discriminators
                    generator_losses['Feature'] += l1_loss(pred_fake[i][j], pred_real[i][j])
            generator_losses['Feature'] *= self.opt.lambda_features/self.opt.num_discriminators

        if not self.opt.no_vgg_loss:
            generator_losses['VGG'] = vgg_loss(real_image, fake_image) * self.opt.lambda_vgg

        return generator_losses, fake_image

    def compute_discriminator_loss(self, segmap_image, real_image):
        """Calculates the different losses related to the discriminator.

             Discriminator hinge loss for the real and fake image.
        """
        discriminator_losses = {}

        fake_image, _ = self.generate_fake(segmap_image, real_image)
        pred_real, pred_fake = self.discriminate(segmap_image, real_image, fake_image)

        discriminator_losses['HingeReal'], discriminator_losses['HingeFake'] = hinge_loss_discriminator(pred_real, pred_fake)

        return discriminator_losses

    def discriminate(self, segmap_image, fake_image, real_image, get_intermediate_features=False):
        """Returns the probability of the real_image and fake_image to be real."""
        #pred_real, pred_fake
        return discriminator(segmap_image, real_image, self.opt.num_discriminators, self.opt.num_discriminator_filters, self.opt.num_discriminator_layers,
                             get_intermediate_features=get_intermediate_features), \
               discriminator(segmap_image, fake_image, self.opt.num_discriminators, self.opt.num_discriminator_filters, self.opt.num_discriminator_layers,
                             get_intermediate_features=get_intermediate_features, reuse=tf.compat.v1.AUTO_REUSE)


    def generate_fake(self, segmap_image, real_image, compute_kld_loss=True):
        """Creates a fake image from the segmentation map of another image."""
        z = None
        KLD_loss = None

        if self.opt.use_vae:
            mean, log_var = encoder(real_image, self.opt.crop_size, self.opt.num_generator_filters)
            z = tf.math.multiply(tf.random.normal(tf.shape(mean)), tf.math.exp(0.5 * log_var)) + mean
            if compute_kld_loss:
                KLD_loss = kld_loss(mean, log_var) * self.opt.lambda_kld

        return generator(segmap_image, self.opt.num_upsampling_layers, z, self.opt.z_dim, self.opt.num_generator_filters), \
               KLD_loss

# def use_gpu():
    # return len(tf.config.list_physical_devices('GPU'))>0
