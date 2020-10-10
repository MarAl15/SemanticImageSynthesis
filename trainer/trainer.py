"""
    Trainer.
"""
import os
import time
import numpy as np
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

        # Save images
        self.save_img_freq = args.save_img_freq
        self.results_dir = args.results_dir

        # Save model
        self.save_model_freq = args.save_model_freq
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_filename = args.checkpoint_filename

        # Training
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.batch_size = args.batch_size

        # Learning rate
        self.lr = tf.compat.v1.placeholder('float', name='learning_rate')
        self.init_lr = args.lr

        # Initialize model
        model = Model(args)

        # Load and shuffle data
        images, _, segmaps_onehot = load_data(args.image_dir, args.label_dir, args.semantic_label_path,
                                     img_size=(args.img_height,args.img_width), crop_size=args.crop_size,
                                     batch_size=args.batch_size, pairing_check=args.pairing_check)
        n_batches = images.cardinality()//args.batch_size
        self.iterations = n_batches.eval()
        photos_and_segmaps = tf.data.Dataset.zip((images, segmaps_onehot)).shuffle(n_batches, reshuffle_each_iteration=True)

        # Define real image and associated one-hot segmentation map
        self.photos_and_segmaps_iterator = tf.compat.v1.data.make_initializable_iterator(photos_and_segmaps)
        self.real, self.segmap = self.photos_and_segmaps_iterator.get_next()

        # Define losses
        self.generator_losses, self.discriminator_losses, self.fake = model.compute_losses(self.segmap, self.real)
        self.generator_loss = tf.math.reduce_sum(list(self.generator_losses.values()))
        self.discriminator_loss = tf.math.reduce_sum(list(self.discriminator_losses.values()))

        # Construct optimizers
        self.generator_optimizer, self.discriminator_optimizer = model.create_optimizers(self.generator_loss, self.discriminator_loss,
                                                                                         self.lr, args.beta1, args.beta2, args.no_TTUR)

        # Define model saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1)

    def train(self):
        # Initialize all variables
        tf.compat.v1.global_variables_initializer().run()

        # Initial learning rate
        lr = self.init_lr

        self.start_time = time.time()

        # Restore (checkpoint) the model if it exits
        checkpoint_step = self.load_model(self.checkpoint_dir)
        if checkpoint_step != -1:
            start_epoch = int(checkpoint_step/self.iterations)
            start_iter =  checkpoint_step - start_epoch*self.iterations + 1
            if start_iter == self.iterations:
                start_epoch += 1
                start_iter = 0
        else:
            start_epoch = 0
            start_iter = 0

        if self.batch_size == 1:
            save_imgs = self.save_img
        for epoch in range(start_epoch, self.epochs):
            self.sess.run(self.photos_and_segmaps_iterator.initializer)
            for batch in range(start_iter, self.iterations):
                real_x, fake_x, _, g_loss, g_losses, _, d_loss, d_losses = self.sess.run([self.real, self.fake, self.generator_optimizer, self.generator_loss, self.generator_losses,
                                                      self.discriminator_optimizer, self.discriminator_loss, self.discriminator_losses],
                                                     feed_dict={self.lr: lr})

                self.print_info(g_loss, g_losses, d_loss, d_losses, epoch+1, batch+1)
                # self.print_info(g_loss, g_losses, d_loss, d_losses, epoch, batch)

                # Save images
                if np.mod(batch+1, self.save_img_freq) == 0:
                    # WARN('Saving images [epoch: %d, step: %d]' % (epoch, batch))
                    # save_imgs(real_x, 'real_image', epoch+1, batch+1)
                    # save_imgs(segmap_x, 'segmentation_map', epoch+1, batch+1)
                    save_imgs(fake_x, 'synthesized_image', epoch+1, batch+1)

                # Save model
                if np.mod(batch+1, self.save_model_freq) == 0:
                    WARN('Saving the model at epoch %d and iteration %d.' % (epoch+1, batch+1))
                    self.save_model(epoch, batch)

            start_iter = 0
            # Save model at the end of the epoch
            self.save_model(epoch, self.iterations)
            WARN('Saving the model at end of the epoch %d.' % (epoch+1))

    def print_info(self, generator_loss, generator_losses, discriminator_loss, discriminator_losses, epoch, iteration):
        msg = "[Epoch: %3d/%3d, Iter: %3d/%3d, Time: %4.4f] - Generator loss: %.3f ~ " % (
                    epoch, self.epochs, iteration, self.iterations, time.time() - self.start_time, generator_loss)

        for key in generator_losses.keys():
            msg += "%s: %.3f, " % (key, generator_losses[key])

        msg +=  "Discriminator loss: %.3f ~ " % (discriminator_loss)
        msg +=  "Fake: %.3f, " % (discriminator_losses['HingeFake'])
        msg +=  "Real: %.3f" % (discriminator_losses['HingeReal'])

        INFO(msg)

    def save_model(self, epoch, iter):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.checkpoint_filename), global_step=self.iterations*epoch+iter)

    def load_model(self, checkpoint_dir):
        print()
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            INFO(" Restoring checkpoint...")

            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return int(ckpt_name.split('-')[-1])

        WARN ("No checkpoint was found.")
        return -1

    def save_img(self, img, type_img, epoch, step):
        img = tf.reshape(img, img.shape[1:])
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        tf.keras.preprocessing.image.save_img(
            '{}/epoch{:03d}_iter{:04d}_{}.png'.format(self.results_dir, epoch, step, type_img),
            img.eval())
