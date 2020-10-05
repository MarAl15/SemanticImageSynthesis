"""
    Trainer.
"""
import time
import tensorflow as tf
from model.model import Model
from utils.load_data import load_data
from utils.pretty_print import *

class Trainer(object):
    """
      Creates the model and optimizers, and uses them to updates the weights of the network while
    reporting losses.
    """
    def __init__(self, sess, args):
        self.sess = sess
        self.opt = args

        # Learning rate
        self.lr = tf.compat.v1.placeholder('float', name='learning_rate')

        # Initialize model
        model = Model(sess, args)

        # Load and shuffle data
        images, _, segmaps_onehot = load_data(args.image_dir, args.label_dir, args.semantic_label_path,
                                     img_size=(args.img_height,args.img_width), crop_size=args.crop_size,
                                     batch_size=args.batch_size, pairing_check=args.pairing_check)
        self.n_batches = images.cardinality()//args.batch_size
        photos_and_segmaps = tf.data.Dataset.zip((images, segmaps_onehot)).shuffle(self.n_batches, reshuffle_each_iteration=True)

        # Define real image and associated one-hot segmentation map
        self.photos_and_segmaps_iterator = tf.compat.v1.data.make_initializable_iterator(photos_and_segmaps)
        self.real, self.segmap = self.photos_and_segmaps_iterator.get_next()

        # Define losses
        generator_losses, self.fake = model.compute_generator_loss(self.segmap, self.real)
        self.generator_loss = tf.math.reduce_sum(list(generator_losses.values()))
        discriminator_losses = model.compute_discriminator_loss(self.segmap, self.real)
        self.discriminator_loss = tf.math.reduce_sum(list(discriminator_losses.values()))

        # Construct optimizers
        self.generator_optimizer, self.discriminator_optimizer = model.create_optimizers(self.generator_loss, self.discriminator_loss)

    def train(self):
        # Initialize all variables
        tf.compat.v1.global_variables_initializer().run()

        lr = self.opt.lr

        start_time = time.time()
        for epoch in range(1, self.opt.epochs+1):
            self.sess.run(self.photos_and_segmaps_iterator.initializer)
            for batch in range(1, self.n_batches.eval()+1):
                _, g_loss, _, d_loss = self.sess.run([self.generator_optimizer, self.generator_loss,
                                                      self.discriminator_optimizer, self.discriminator_loss],
                                                     feed_dict={self.lr: lr})

                INFO("Epoch: [%3d] [%3d/%3d] Time: %4.4f - Discriminator loss: %.8f, Generator loss: %.8f" % (
                    epoch, batch, self.n_batches.eval(), time.time() - start_time, d_loss, g_loss))
