"""
    SPADE Architecture.
"""
import tensorflow as tf
from utils.utils import Conv2d, leaky_relu
from model.networks.normalizations import spade


def spade_resblk(segmap, x, k_out, spectral_norm=False):
    """
        Residual block.

          Takes in the segmentation map (|segmap|) as input, learns the skip connection if necessary, and applies
        normalization first and then convolution.
    """
    k_in = x.shape[-1]
    k_middle = min(k_in, k_out)

    with tf.compat.v1.variable_scope('ResBlk'):
        dx = spade(segmap, x, k_in)
        dx = leaky_relu(dx)
        dx = Conv2d(dx, k_middle, kernel_size=3, padding=1, spectral_norm=spectral_norm)

        dx = spade(segmap, dx, k_middle)
        dx = leaky_relu(dx)
        dx = Conv2d(dx, k_out, kernel_size=3, padding=1, spectral_norm=spectral_norm)

        if k_in != k_out:
            x_s = spade(segmap, x, k_in)
            # x_s = leaky_relu(x_s)
            x_s = Conv2d(x_s, k_out, kernel_size=1, spectral_norm=spectral_norm, use_bias=False)

            return x_s + dx

        return x + dx


def VGG19(x, trainable=False, slices=None):
    if slices == None:
        slices = initialize_vgg19(trainable)

    h_relu1 = slices[0](x)
    h_relu2 = slices[1](h_relu1)
    h_relu3 = slices[2](h_relu2)
    h_relu4 = slices[3](h_relu3)

    return [h_relu1, h_relu2, h_relu3, h_relu4, slices[4](h_relu4)]

def initialize_vgg19(trainable=False):
    model = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False)

    if not trainable:
        model.trainable = False

    vgg_pretrained_features = model.layers

    slice1 = tf.keras.Sequential()
    slice2 = tf.keras.Sequential()
    slice3 = tf.keras.Sequential()
    slice4 = tf.keras.Sequential()
    slice5 = tf.keras.Sequential()

    # for i, layer in enumerate(vgg_pretrained_features):
        # print(str(i)+') '+layer.name)

    # block1_conv1
    for i in range(1, 2):
        slice1.add(vgg_pretrained_features[i])

    # block1_conv2, block1_pool, block2_conv1
    for i in range(2, 5):
        slice2.add(vgg_pretrained_features[i])

    # block2_conv2, block2_pool, block3_conv1
    for i in range(5, 8):
        slice3.add(vgg_pretrained_features[i])

    # block3_conv2, block3_conv3, block3_conv4, block3_pool, block4_conv1
    for i in range(8, 13):
        slice4.add(vgg_pretrained_features[i])

    # block4_conv2, block4_conv3, block4_conv4, block4_pool, block5_conv1
    for i in range(13, 18):
        slice5.add(vgg_pretrained_features[i])

    return [slice1, slice2, slice3, slice4, slice5]
