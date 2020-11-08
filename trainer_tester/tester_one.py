"""
    Generate an image.
"""
import os
import tensorflow as tf
from model.model import Model
from model.networks.encoder import Encoder
from model.networks.generator import Generator
from utils.load_data import get_all_labels, change_labels
from utils.pretty_print import *


class TesterOne(object):
    def __init__(self, args):
        # Save images
        self.results_dir = args.results_dir

        # Preprocess image
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.crop_size = args.crop_size
        self.labels = get_all_labels(None, args.semantic_label_path)
        self.n_labels = len(self.labels)

        # Use VAE
        self.use_vae = args.use_vae

        # Define Encoder, Generator
        img_shape = [1, args.crop_size, args.crop_size, 3]
        segmap_shape = [1, args.crop_size, args.crop_size, self.n_labels]
        if args.use_vae:
            encoder = Encoder(img_shape, crop_size=args.crop_size, num_filters=args.num_encoder_filters)
        generator = Generator(segmap_shape, num_upsampling_layers=args.num_upsampling_layers,
                              num_filters=args.num_generator_filters, use_vae=args.use_vae)

        # Initialize model
        self.model = Model(args, generator, None, encoder, training=False) if args.use_vae else \
                     Model(args, generator, None, training=False)

        # Define checkpoint-saver
        self.checkpoint = tf.train.Checkpoint(encoder=encoder,
                                              generator=generator) if args.use_vae else \
                          tf.train.Checkpoint(generator=generator)
        self.manager_model = tf.train.CheckpointManager(self.checkpoint, args.checkpoint_dir, max_to_keep=3)

        # Restore the latest checkpoint in checkpoint_dir
        self.checkpoint.restore(self.manager_model.latest_checkpoint)
        print()
        if self.manager_model.latest_checkpoint:
            INFO("Checkpoint restored from " + self.manager_model.latest_checkpoint)
        else:
            ERROR("No checkpoint was found.")
        print()

    def generate_fake(self, segmap_path, styimg_path=None):
        # Load the segmentation map
        segmap = self.preprocess_image(segmap_path, num_channels=1,
                                       resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                       normalize=False)

        segmap = change_labels(segmap, self.labels)
        print(segmap)

        # Transform the segmentation map to one-hot encoding
        segmap = tf.one_hot(tf.squeeze(segmap, -1), self.n_labels)

        # Generate fake image
        if self.use_vae:
            # Load the style image
            style_img = self.preprocess_image(styimg_path)

            fake_image, _ = self.model.generate_fake(segmap, style_img)
        else:
            fake_image, _ = self.model.generate_fake(segmap)

        # Save fake image
        filename, _ = os.path.splitext(os.path.split(segmap_path)[1])
        self.save_img(fake_image, filename+"_synthesized.png")


    def preprocess_image(self, image_file, num_channels=3, resize_method=tf.image.ResizeMethod.BICUBIC, normalize=True):
        # Load image
        img = tf.io.read_file(image_file)
        img = tf.image.decode_jpeg(img, channels=num_channels)

        # Resize
        img = tf.image.resize(img, (self.img_height, self.img_width),
                              method=resize_method)

        # Reshape
        img = tf.reshape(img, [1, self.img_height, self.img_width, num_channels])

        # Crop
        img = tf.image.random_crop(img, size=[1, self.crop_size, self.crop_size, num_channels])

        if (normalize):
            # Normalize
            img = tf.cast(img, 'float') / 127.5 - 1

        _, filename_with_extension = os.path.split(image_file)

        # Save image
        self.save_img(img, filename_with_extension)

        return img

    def save_img(self, image, filename):
        """Saves the given image."""
        img = image[0]
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        tf.keras.preprocessing.image.save_img('{}/{}'.format(self.results_dir, filename),
                                              img)
