"""
    Builds and trains the Semantic Image Synthesis model, in addition to loading the data.

    Author: Mar Alguacil
"""
import argparse
from utils.pretty_print import *
from utils.load_data import load_data

import tensorflow as tf


def parse_args():
    """Particular configuration"""
    parser = argparse.ArgumentParser(description="Semantic Image Synthesis")

    #################
    #   LOAD DATA   #
    #################
    # CHECK IMAGE-LABEL FILE PAIRING
    parser.add_argument('--pairing_check', dest='pairing_check', action='store_true',
                        help='If specified, check of correct image-label file pairing ' + INFO_COLOR('(by default).'))
    parser.add_argument('--no_pairing_check', dest='pairing_check', action='store_false',
                        help='If specified, skip sanity check of correct image-label file pairing.')
    parser.set_defaults(pairing_check=True)

    # LOAD DATA FROM FOLDERS
    #  If the error "Input 'filename' of 'ReadFile' Op has type float32 that does not match expected type of string." is thrown,
    # create new subdirectories to store them in. For instance,
    #     (Directory structure)
    #     img_train_path/
    #     ...train/
    #     ......train_image_001.jpg
    #     ......train_image_002.jpg
    #     ...... ...
    #     segmask_train_path/
    #     ...train/
    #     ......label_image_001.jpg
    #     ......label_image_002.jpg
    #     ...... ...
    # =>
    #     --image_dir img_train_path
    #     --label_dir segmask_train_path
    parser.add_argument('--image_dir', type=str, default='./datasets/ADEChallengeData2016/images/training',
                           help="Main directory name where the pictures are located. " +
                                 INFO_COLOR("Default: './datasets/ADEChallengeData2016/images/training'"))
    parser.add_argument('--label_dir', type=str, default='./datasets/ADEChallengeData2016/annotations/training',
                           help="Main directory name where the semantic segmentation masks are located. " +
                                 INFO_COLOR("Default: './datasets/ADEChallengeData2016/annotations/training'"))
    parser.add_argument('--semantic_label_path', type=str, default='./datasets/ADEChallengeData2016/semantic_labels.txt',
                           help="Filename containing the semantic labels. " +
                                 INFO_COLOR("Default: './datasets/ADEChallengeData2016/semantic_labels.txt'"))

    # RESIZE IMAGES
    parser.add_argument('--img_height', type=int, default=286,
                           help='The height size of image. ' + INFO_COLOR('Default: 286'))
    parser.add_argument('--img_width', type=int, default=286,
                           help='The width size of image. ' + INFO_COLOR('Default: 286'))

    # CROP IMAGES
    parser.add_argument('--crop_size', type=int, default=256,
                           help='Desired size of the square crop. ' + INFO_COLOR('Default: 256'))

    # BATCHES
    parser.add_argument('--batch_size', type=int, default=1,
                           help='Input batch size. ' + INFO_COLOR('Default: 1'))


    #################
    #   GENERATOR   #
    #################
    parser.add_argument('--num_upsampling_layers', choices=('normal', 'more', 'most'), default='more',
                           help="If 'more', adds upsampling layer after the second resnet block. \
                                 If 'most', also adds one more upsampling + resnet layer after the last resnet block. " +
                                 INFO_COLOR("Default: 'more'"))
    parser.add_argument('--z_dim', type=int, default=256,
                           help='Dimension of the latent z vector. ' + INFO_COLOR('Default: 256'))
    parser.add_argument('--num_generator_filters', type=int, default=64,
                           help='Number of generator filters in penultimate convolutional layers. ' +
                                 INFO_COLOR('Default: 64'))

    #####################
    #   DISCRIMINATOR   #
    #####################
    parser.add_argument('--num_discriminators', type=int, default=2,
                           help='Number of discriminators to be used in multiscale. ' + INFO_COLOR("Default: 2"))
    parser.add_argument('--num_discriminator_layers', type=int, default=4,
                           help='Number of layers in each discriminator. ' + INFO_COLOR('Default: 4'))
    parser.add_argument('--num_discriminator_filters', type=int, default=64,
                           help='Number of discrimator filters in first convolutional layer. ' +
                                 INFO_COLOR('Default: 64'))


    return parser.parse_args()

def main():
    gpus= tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # Parse arguments
    args = parse_args()

    # ~ with tf.device("/cpu:0"):
    # Load data from folders
    images, labels = load_data(args.image_dir, args.label_dir, img_size=(args.img_height,args.img_width), crop_size=args.crop_size,
                               batch_size=args.batch_size, pairing_check=args.pairing_check)


if __name__ == '__main__':
    main()
