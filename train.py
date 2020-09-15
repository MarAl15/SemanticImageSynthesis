"""
    Loads the data, in addition to building and training the model.

    Author: Mar Alguacil
"""
import argparse
from utils.load_data import *

def parse_args():
    """Particular configuration"""
    parser = argparse.ArgumentParser(description="Semantic Image Synthesis")

    #################
    #   LOAD DATA   #
    #################
    # CHECK IMAGE-LABEL FILE PAIRING
    parser.add_argument('--pairing_check', dest='pairing_check', action='store_true',
                        help='If specified, check of correct image-label file pairing (by default).')
    parser.add_argument('--no_pairing_check', dest='pairing_check', action='store_false',
                        help='If specified, skip sanity check of correct image-label file pairing.')
    parser.set_defaults(pairing_check=True)

    # LOAD DATA FROM FOLDERS
    #  If the error "Input 'filename' of 'ReadFile' Op has type float32 that does not match expected type of string." is thrown,
    # put all images in a new subdirectory.
    # E.g.:
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
                        help='Directory name where the pictures are located.')
    parser.add_argument('--label_dir', type=str, default='./datasets/ADEChallengeData2016/annotations/training',
                        help='Directory name where the semantic segmentation masks are located.')

    # RESIZE IMAGES
    parser.add_argument('--img_height', type=int, default=286, help='The height size of image.')
    parser.add_argument('--img_width', type=int, default=286, help='The width size of image. ')

    # CROP IMAGES
    parser.add_argument('--crop_size', type=int, default=256, help='Crop to the width of crop_size.')

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # ~ with tf.device("/cpu:0"):
    load_data(args.image_dir, args.label_dir, img_size=(args.img_height,args.img_width), crop_size=args.crop_size, pairing_check=args.pairing_check)



if __name__ == '__main__':
    main()
