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
from utils.load_data import load_data
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

        # Print info
        self.print_info_freq = args.print_info_freq

        # Training
        self.total_epochs = args.total_epochs

        # Learning rate
        self.lr = args.lr
        self.decay_epoch = args.decay_epoch
        self.no_TTUR = args.no_TTUR
        self.lrd = args.lr/(self.total_epochs-self.decay_epoch)

        # Load and shuffle data
        images, segmaps, self.n_labels = load_data(args.image_dir, args.label_dir, args.semantic_label_path,
                                    img_size=(args.img_height,args.img_width), crop_size=args.crop_size,
                                    batch_size=args.batch_size, pairing_check=args.pairing_check)
        self.iterations = int(tf.constant(args.prob_dataset)*tf.cast(images.cardinality()/args.batch_size, 'float'))
        self.photos_and_segmaps = tf.data.Dataset.zip((images, segmaps)).shuffle(self.iterations, reshuffle_each_iteration=True)
        # self.photos_and_segmaps = tf.data.Dataset.zip((images, segmaps))

        # Define Encoder, Generator, Discriminator
        img_shape = [args.batch_size, args.crop_size, args.crop_size, 3]
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

        # Define checkpoint-saver
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                              lr=tf.Variable(self.lr),
                                              generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              encoder=encoder,
                                              generator=generator,
                                              discriminator=discriminator) if args.use_vae else \
                          tf.train.Checkpoint(step=tf.Variable(1),
                                              lr=tf.Variable(self.lr),
                                              generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=generator,
                                              discriminator=discriminator)
        self.manager_model = tf.train.CheckpointManager(self.checkpoint, args.checkpoint_dir, max_to_keep=3)

        # Define summary file writer to log losses
        self.summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, "fit/"))


    def fit(self):
        """Trains the model for a fixed number of epochs."""
        # Restore the latest checkpoint in checkpoint_dir
        self.checkpoint.restore(self.manager_model.latest_checkpoint)
        print()
        if self.manager_model.latest_checkpoint:
            INFO("Checkpoint restored from " + self.manager_model.latest_checkpoint)
            start_epoch = int(self.checkpoint.step // self.iterations) + 1
            start_iter = int(self.checkpoint.step % self.iterations) + 1
            self.checkpoint.step.assign_add(1)
        else:
            WARN("No checkpoint was found. Initializing from scratch.")
            start_epoch, start_iter = 1, 1
        print()

        for epoch in range(start_epoch, self.total_epochs+1):
            start_time = time.time()
            if epoch>=self.decay_epoch and start_iter==1:
                self.update_learning_rate()

            # Train
            for n, (real_image, segmap) in self.photos_and_segmaps.take(self.iterations-start_iter+1).enumerate(start=start_iter):
                # With `clear_session()` called at the beginning, Keras starts with a blank state at each iteration
                # and memory consumption is constant over time.
                tf.keras.backend.clear_session()

                # Train
                fake_image = self.train_step(real_image, segmap, epoch, n)

                # Force the summary writer to write to disk
                self.summary_writer.flush()

                # Save (checkpoint) model every self.save_model_freq steps
                if (self.checkpoint.step % self.save_model_freq) == 0:
                    INFO('Saving model at epoch %d and iteration %d...' % (epoch, n))
                    self.manager_model.save()

                # Save image every self.save_img_freq steps
                if (self.checkpoint.step % self.save_img_freq) == 0:
                    self.save_img(real_image, 'real_image', epoch, n)
                    self.save_img(segmap, 'segmentation_map', epoch, n)
                    self.save_img(fake_image, 'synthesized_image', epoch, n)

                self.checkpoint.step.assign_add(1)

            INFO('Time taken for epoch %d is %.3f sec' % (epoch, time.time()-start_time))


            # Save (checkpoint) model at the end of each epoch
            INFO('Saving model at end of the epoch %d...' % epoch)
            self.manager_model.save()

            start_iter = 1

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

            # Display information every self.print_info_freq steps
            if (self.checkpoint.step % self.print_info_freq) == 0:
                self.print_info(total_generator_loss, generator_losses, total_discriminator_loss, discriminator_losses, epoch, iteration)

            # Log the losses to TensorBoard
            with self.summary_writer.as_default():
                tf.summary.scalar('gen_total_loss', total_generator_loss, step=epoch)
                for key in generator_losses.keys():
                    tf.summary.scalar('gen_'+key.lower()+'_loss', generator_losses[key], step=epoch)

                tf.summary.scalar('disc_total_loss', total_discriminator_loss, step=epoch)
                tf.summary.scalar('disc_fake_loss', discriminator_losses['HingeFake'], step=epoch)
                tf.summary.scalar('disc_real_loss', discriminator_losses['HingeReal'], step=epoch)


            return fake_image

    def update_learning_rate(self):
        """Applies linear decay to the learning rate."""
        self.lr = self.lr - self.lrd
        (g_lr, d_lr) = (self.lr/2, self.lr*2) if not self.no_TTUR else (self.lr, self.lr)

        self.generator_optimizer.lr.assign(g_lr)
        self.discriminator_optimizer.lr.assign(d_lr)


    def print_info(self, generator_loss, generator_losses, discriminator_loss, discriminator_losses, epoch, iteration):
        """Displays information."""
        tf.print("\033[22;33m[Epoch: %3d/%d, Iter: " % (epoch, self.total_epochs), iteration, "/%d] \033[0m" % self.iterations, end=' ', sep='')

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


    def save_img(self, img_batch, type_img, epoch, iteration):
        """Saves the first image of the batch."""
        img = img_batch[0]
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        tf.keras.preprocessing.image.save_img('{}/epoch{:03d}_iter{:05d}_{}.png'.format(self.results_dir, epoch, iteration, type_img),
                                              img)
