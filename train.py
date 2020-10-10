"""
    Train.
"""
import tensorflow as tf
from utils.pretty_print import *
from trainer.trainer import Trainer
from trainer.options import parse_args


def main():
    # Parse arguments
    args = parse_args()

    # ~ with tf.device("/cpu:0"):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        # Initialize trainer
        trainer = Trainer(sess, args)

        # Train
        trainer.train()




if __name__ == '__main__':
    main()
