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
        self.num_discriminators = args.num_discriminators

        # Loss weights
        if self.use_vae:
            self.lambda_kld = args.lambda_kld
        # To use or not discriminator feature matching loss
        self.no_feature_loss = args.no_feature_loss
        if not self.no_feature_loss:
            self.lambda_features = args.lambda_features
        # To use or not VGG loss
        self.no_vgg_loss = args.no_vgg_loss
        if not self.no_vgg_loss:
            self.lambda_vgg = args.lambda_vgg
            self.vgg_loss = VGGLoss([args.batch_size, args.crop_size, args.crop_size, 3])

        # Training?
        self.training = training

    def create_optimizers(self, lr, beta1, beta2, no_TTUR=True):
        """Constructs the generator and discriminator optimizers."""
        (g_lr, d_lr) = (lr/2, lr*2) if not no_TTUR else (lr, lr)

        # generator_optimizer, discriminator_optimizer
        return tf.keras.optimizers.Adam(g_lr, beta_1=beta1, beta_2=beta2), \
               tf.keras.optimizers.Adam(d_lr, beta_1=beta1, beta_2=beta2)


    def compute_generator_loss(self, pred_real, pred_fake, real_image, fake_image, mean_var=None):
        """Calculates the different losses related to the generator.

            Generator hinge loss for the fake image.
            If use_vae is specified, KLD loss.
            If no_feature_loss is not specified, average L1 loss of each intermediate output.
            If no_vgg_loss is not specified, VGG loss.
        """
        generator_losses = {}

        generator_losses['Hinge'] = hinge_loss_generator(pred_fake)

        if self.use_vae:
            generator_losses['KLD'] = kld_loss(mean_var[0], mean_var[1]) * self.lambda_kld

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
            generator_losses['VGG'] = self.vgg_loss(real_image, fake_image) * self.lambda_vgg

        return tf.math.reduce_sum(list(generator_losses.values())), generator_losses,

    def compute_discriminator_loss(self, pred_real, pred_fake):
        """Calculates the different losses related to the discriminator.

            Discriminator hinge loss for the real and fake image.
        """
        discriminator_losses = {}

        discriminator_losses['HingeReal'], discriminator_losses['HingeFake'] = hinge_loss_discriminator(pred_real, pred_fake)

        return discriminator_losses['HingeReal']+discriminator_losses['HingeFake'], discriminator_losses


    def discriminate(self, real_image, segmap_image, fake_image):
        """Returns the prediction of the real and fake image."""
        #pred_real, pred_fake
        return self.discriminator([real_image, segmap_image], training=True), \
               self.discriminator([fake_image, segmap_image], training=True)


    def generate_fake(self, segmap_image, real_image=None):
        """Creates a fake image from the segmentation map of another image."""
        if self.use_vae:
            mean, log_var = self.encoder(real_image, training=True)
            z = tf.math.multiply(tf.random.normal(mean.shape), tf.math.exp(0.5 * log_var)) + mean

            return self.generator([segmap_image, z]), [mean, log_var]

        return self.generator(segmap_image, training=True), None


# def use_gpu():
    # return len(tf.config.list_physical_devices('GPU'))>0
