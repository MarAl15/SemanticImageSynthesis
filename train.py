"""
    Train.
"""
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from utils.pretty_print import *
from trainer_tester.trainer import Trainer
from trainer_tester.options import parse_args


def main():
    gpus= tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # Parse arguments
    args = parse_args()

    # ~ with tf.device("/cpu:0"):

    # Initialize trainer
    trainer = Trainer(args)

    # Train
    trainer.fit()


if __name__ == '__main__':
    main()
