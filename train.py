"""
    Train.
"""
import tensorflow as tf
from utils.pretty_print import *
from trainer.trainer import Trainer
from trainer.options import parse_args


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
