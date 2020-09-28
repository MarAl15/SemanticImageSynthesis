"""
    Loads the data.

    Author: Mar Alguacil
"""
import os
import pathlib
import numpy as np
from glob import glob
from ast import literal_eval
from utils.pretty_print import *

import tensorflow as tf


def load_data(image_folder, label_folder, img_size=(286,286), resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, crop_size=256, batch_size=1, pairing_check=True):
    """
        Loads the images and segmentation masks.
    """
    assert os.path.isdir(image_folder), ERROR_COLOR('%s is not a valid directory'%image_folder)
    assert os.path.isdir(label_folder), ERROR_COLOR('%s is not a valid directory'%label_folder)

    if pairing_check:
        INFO("Checking of correct image-label file pairing...")
        image_paths = sorted(np.array(glob(os.path.join(image_folder, '*.*'))))
        if not image_paths:
            image_paths = sorted(np.array(glob(os.path.join(image_folder, '*/*.*'))))

        label_paths = sorted(np.array(glob(os.path.join(label_folder, '*.*'))))
        if not label_paths:
            label_paths = sorted(np.array(glob(os.path.join(label_folder, '*/*.*'))))
        for img_path, label_path in zip(image_paths, label_paths):
            assert files_match(img_path, label_path), \
                    ERROR_COLOR("The image-label pair (%s, %s) does not seem to be linked. The filenames are different." % (img_path, label_path))

    print()
    INFO("Loading images...")
    images = load_images(image_folder, img_size, crop_size, resize_method=resize_method, color_mode='rgb', batch_size=batch_size, normalize=True)

    print()
    INFO("Loading segmentation masks...")
    labels = load_images(label_folder, img_size, crop_size, resize_method=resize_method, color_mode='grayscale', batch_size=batch_size, normalize=False)

    return images, labels



def files_match(path1, path2):
    """
        Checks if the filename1 (from path1) and filename2 (from path2) without extension are the same.
    """
    return os.path.splitext(os.path.basename(path1))[0] == os.path.splitext(os.path.basename(path2))[0]


def load_images(folder, img_size, crop_size, resize_method=tf.image.ResizeMethod.BICUBIC, color_mode='rgb', batch_size=1, normalize=True):
    """
        Loads, resizes and crops the images from the specified folder.
        If normalize is True, the images are also rescaled to [0,1].
    """
    INFO(" - Resizing...")
    images = tf.keras.preprocessing.image_dataset_from_directory(folder,
                                                                 label_mode = None,
                                                                 color_mode = color_mode,
                                                                 image_size = img_size,
                                                                 interpolation = resize_method,
                                                                 batch_size=batch_size,
                                                                 shuffle=False)

    INFO(" - Cropping...")
    images = images.map(lambda img: tf.image.random_crop(img, size=[tf.shape(img)[0], crop_size, crop_size, tf.shape(img)[3]]), num_parallel_calls=12)

    if (normalize):
        INFO(" - Standardizing...")
        #  The RGB channel values are in the [0, 255] range. This is not ideal for a neural network.
        # For this reason, we will standardize values to be in the [0, 1] by using a Rescaling layer.
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        images = images.map(normalization_layer, num_parallel_calls=12)

    return images


def get_all_labels(segmap_dataset, semantic_label_path):
    """
        Extracts semantic labels.
    """
    if os.path.exists(semantic_label_path) :
        with open(semantic_label_path, 'r') as f:
            labels = literal_eval(f.read())
    else:
        first = True

        for segmap in segmap_dataset:
            segmap = tf.reshape(segmap, [-1])
            segmap_labels, _ = tf.unique(segmap)

            if not first:
                for x in segmap_labels:
                    if x not in labels:
                        labels = np.append(labels, x)
            else:
                labels = segmap_labels.numpy()
                first = False

        with open(semantic_label_path, 'w') as f :
            f.write(", ".join(str(v) for v in labels))

    return labels
