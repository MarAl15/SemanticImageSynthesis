"""
    Trainer.
"""
import os
import time
import numpy as np
import tensorflow as tf
from model.model import Model
from model.networks.encoder import encoder
from model.networks.generator import generator
from model.networks.discriminator import discriminator
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
        images, segmaps, segmaps_onehot = load_data(args.image_dir, args.label_dir, args.semantic_label_path,
                                     img_size=(args.img_height,args.img_width), crop_size=args.crop_size,
                                     batch_size=args.batch_size, pairing_check=args.pairing_check)
        self.iterations = images.cardinality()//args.batch_size
        self.photos_and_segmaps = tf.data.Dataset.zip((images, segmaps_onehot)).shuffle(self.iterations, reshuffle_each_iteration=True)

        # Define Encoder, Generator, Discriminator
        img_shape = [args.batch_size, args.crop_size, args.crop_size, 3]
        n_labels = len(get_all_labels(segmaps, args.semantic_label_path))
        segmap_shape = [args.batch_size, args.crop_size, args.crop_size, n_labels]
        if args.use_vae:
            self.encoder = encoder(img_shape, crop_size=args.crop_size, num_filters=args.num_encoder_filters)
        self.generator = generator(segmap_shape, num_upsampling_layers=args.num_upsampling_layers,
                                   num_filters=args.num_generator_filters, use_vae=args.use_vae)
        self.discriminator = discriminator(img_shape, segmap_shape, num_discriminators=args.num_discriminators,
                                           num_filters=args.num_discriminator_filters,
                                           num_layers=args.num_discriminator_layers,
                                           get_intermediate_features=(not args.no_feature_loss))

        # Initialize model
        model = Model(args, self.generator, self.discriminator, self.encoder, training=True) if args.use_vae else \
                Model(args, self.generator, self.discriminator, training=True)

        # Define losses
        # self.generator_losses, self.discriminator_losses, self.fake = model.compute_losses(self.segmap, self.real)
        # self.generator_loss = tf.math.reduce_sum(list(self.generator_losses.values()))
        # self.discriminator_loss = tf.math.reduce_sum(list(self.discriminator_losses.values()))

        # Construct optimizers
        # self.generator_optimizer, self.discriminator_optimizer = model.create_optimizers(self.generator_loss, self.discriminator_loss,
                                                                                         # self.lr, args.beta1, args.beta2, args.no_TTUR)

        # Define model saver


    def print_info(self, generator_loss, generator_losses, discriminator_loss, discriminator_losses, epoch, iteration):
        msg = "[Epoch: %3d/%3d, Iter: %3d/%3d, Time: %4.4f] - Generator loss: %.3f ~ " % (
                    epoch, self.epochs, iteration, self.iterations, time.time() - self.start_time, generator_loss)

        for key in generator_losses.keys():
            msg += "%s: %.3f, " % (key, generator_losses[key])

        msg +=  "Discriminator loss: %.3f ~ " % (discriminator_loss)
        msg +=  "Fake: %.3f, " % (discriminator_losses['HingeFake'])
        msg +=  "Real: %.3f" % (discriminator_losses['HingeReal'])

        INFO(msg)


    def save_img(self, img, type_img, epoch, step):
        img = tf.reshape(img, img.shape[1:])
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        tf.keras.preprocessing.image.save_img(
            '{}/epoch{:03d}_iter{:04d}_{}.png'.format(self.results_dir, epoch, step, type_img),
            img)
