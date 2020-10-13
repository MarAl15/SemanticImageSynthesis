"""
    Trainer.
"""
import os
import time
import numpy as np
import tensorflow as tf
from model.model import Model
from model.networks.encoder import Encoder
from model.networks.generator import Generator
from model.networks.discriminator import Discriminator
from utils.load_data import load_data, get_all_labels
from utils.pretty_print import *


class Trainer(object):
    """
      Creates the model and optimizers, and uses them to updates the weights of the network while
    reporting losses.
    """
    def __init__(self, args):
        # Save images
        self.save_img_freq = args.save_img_freq
        self.results_dir = args.results_dir

        # Save model
        self.save_model_freq = args.save_model_freq
        self.checkpoint_dir = args.checkpoint_dir

        # Training
        self.epochs = args.epochs

        # Learning rate
        self.lr = args.lr

        # Load and shuffle data
        images, segmaps = load_data(args.image_dir, args.label_dir,
                                    img_size=(args.img_height,args.img_width), crop_size=args.crop_size,
                                    batch_size=args.batch_size, pairing_check=args.pairing_check)
        self.iterations = int(tf.constant(args.prob_dataset)*tf.cast(images.cardinality()/args.batch_size, 'float'))
        self.photos_and_segmaps = tf.data.Dataset.zip((images, segmaps)).shuffle(self.iterations, reshuffle_each_iteration=True)

        # Define Encoder, Generator, Discriminator
        img_shape = [args.batch_size, args.crop_size, args.crop_size, 3]
        self.n_labels = len(get_all_labels(segmaps, args.semantic_label_path))
        segmap_shape = [args.batch_size, args.crop_size, args.crop_size, self.n_labels]
        if args.use_vae:
            encoder = Encoder(img_shape, crop_size=args.crop_size, num_filters=args.num_encoder_filters)
        generator = Generator(segmap_shape, num_upsampling_layers=args.num_upsampling_layers,
                              num_filters=args.num_generator_filters, use_vae=args.use_vae)
        discriminator = Discriminator(img_shape, segmap_shape, num_discriminators=args.num_discriminators,
                                      num_filters=args.num_discriminator_filters,
                                      num_layers=args.num_discriminator_layers,
                                      get_intermediate_features=(not args.no_feature_loss))

        # Initialize model
        self.model = Model(args, generator, discriminator, encoder, training=True) if args.use_vae else \
                     Model(args, generator, discriminator, training=True)

        # Construct optimizers
        self.generator_optimizer, self.discriminator_optimizer = self.model.create_optimizers(self.lr, args.beta1, args.beta2, args.no_TTUR)

        # Trainable variables
        self.generator_vars = generator.trainable_variables
        if args.use_vae:
            self.generator_vars += encoder.trainable_variables
        self.discriminator_vars = discriminator.trainable_variables


    @tf.function
    def train_step(self, real_image, segmap, epoch, iteration):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Transform the segmentation map to one-hot encoding
            segmap = tf.one_hot(tf.squeeze(segmap, -1), self.n_labels)

            # Generate fake image
            fake_image, mean_var = self.model.generate_fake(segmap, real_image)
            # Discriminator predictions
            pred_real, pred_fake = self.model.discriminate(real_image, segmap, fake_image)

            # Losses
            total_generator_loss, generator_losses = self.model.compute_generator_loss(pred_real, pred_fake, real_image, fake_image, mean_var)
            total_discriminator_loss, discriminator_losses = self.model.compute_discriminator_loss(pred_real, pred_fake)

            # Gradients
            generator_gradients = gen_tape.gradient(total_generator_loss, self.generator_vars)
            discriminator_gradients = disc_tape.gradient(total_discriminator_loss, self.discriminator_vars)

            # Optimizers
            self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator_vars))
            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator_vars))

            # Display information
            self.print_info(total_generator_loss, generator_losses, total_discriminator_loss, discriminator_losses, epoch, iteration)


    def print_info(self, generator_loss, generator_losses, discriminator_loss, discriminator_losses, epoch, iteration):
        tf.print("\033[22;33m[Epoch: %3d/%d, Iter: " % (epoch, self.epochs), iteration, "/%d," % self.iterations,
                             " Time: %.3f] \033[0m" % (time.time() - self.start_time), end=' ', sep='')

        tf.print("\033[01;34mGenerator loss:\033[22;34m", end='')
        tf_print_float(generator_loss)
        tf.print('\033[22;32m -', end=' ')
        for key in generator_losses.keys():
            tf.print(key, ":", end='', sep='')
            tf_print_float(generator_losses[key])
            tf.print(', ', end='')

        tf.print("~\033[01;34m Discriminator loss:\033[22;34m", end='')
        tf_print_float(discriminator_loss)
        tf.print('\033[22;32m -', end=' ')
        tf.print("Fake:", end='')
        tf_print_float(discriminator_losses['HingeFake'])
        tf.print(',', end=' ')
        tf.print("Real:", end='')
        tf_print_float(discriminator_losses['HingeReal'])
        tf.print('\033[0m', sep='')


    def save_img(self, img, type_img, epoch, step):
        img = tf.reshape(img, img.shape[1:])
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        tf.keras.preprocessing.image.save_img(
            '{}/epoch{:03d}_iter{:04d}_{}.png'.format(self.results_dir, epoch, step, type_img),
            img)
