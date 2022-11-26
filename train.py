"""
    Train.
"""

########################################################################
## Uncomment if you want to disable Tensorflow debugging information. ##
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
########################################################################
import tensorflow as tf
from utils.pretty_print import *
from trainer_tester.trainer import Trainer
from trainer_tester.options import parse_args


def main():
    # Parse arguments
    args = parse_args()

    # Enable memory growth for a GPU
    gpus= tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        WARN("Invalid device or cannot modify virtual devices once initialized.")
        pass

    # Initialize trainer
    trainer = Trainer(args)

    # Train
    trainer.fit()


if __name__ == '__main__':
    main()