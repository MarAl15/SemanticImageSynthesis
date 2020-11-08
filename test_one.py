"""
    Generate and save a fake image.
"""
import logging
import tensorflow as tf
from utils.pretty_print import *
from trainer_tester.tester_one import TesterOne
from trainer_tester.options import parse_args


def main():
    tf.get_logger().setLevel(logging.ERROR)
    gpus= tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # Parse arguments
    args = parse_args(train=False, one=True)

    # Initialize tester
    tester = TesterOne(args)

    # Generate and save fake image
    tester.generate_fake(args.label_filename, args.style_filename)



if __name__ == '__main__':
    main()
