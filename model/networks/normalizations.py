"""
    Normalizations.
"""
import tensorflow as tf
import tensorflow_addons as tfa
from utils.utils import Conv2d, static_vars


def spade(segmap, x, k_filters, nhidden=128, kernel_size=3):
    """Applies SPatially-Adaptive (DE)normalization (SPADE).

          In the SPADE, the segmentation map (|segmap|) is first projected onto an embedding space and then convolved to
        produce the modulation parameters |alpha| and |beta|. The produced |alpha| and |beta| are multiplied and added to
        the normalized activation element-wise (Park et al.).
    """
    with tf.compat.v1.variable_scope('SPADE'):
        # Generate normalized activations
        normalized = batch_normalization(x)

        # Produce scaling and bias conditioned on semantic map
        segmap = tf.image.resize(images=segmap, size=tf.shape(x)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        pw = kernel_size//2
        actv = Conv2d(segmap, nhidden, kernel_size, padding=pw)
        actv = tf.nn.relu(actv)

        gamma = Conv2d(actv, k_filters, kernel_size, padding=pw)
        beta = Conv2d(actv, k_filters, kernel_size, padding=pw)


        return normalized * (1 + gamma) + beta


def batch_normalization(x, epsilon=1e-05):
    """"Applies Batch Normalization."""
    mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)

    # epsilon -> A small float number to avoid dividing by 0.
    return (x - mean) / tf.sqrt(variance + epsilon)


@static_vars(i=0)
def spectral_normalization(w, n_power_iterations=1):
    """Applies Spectral Normalization.

        Simple Tensorflow Implementation of Spectral Normalization for Generative Adversarial Networks (ICLR 2018).
        Taken from https://github.com/taki0112/Spectral_Normalization-Tensorflow.
    """
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.compat.v1.get_variable("u"+str(spectral_normalization.i), [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
    spectral_normalization.i+=1
    # u = tf.random_normal_initializer()(shape=[1, w_shape[-1]])

    u_hat = u
    v_hat = None
    for i in range(n_power_iterations):
       # Power iteration
       # Usually n_power_iterations = 1 will be enough

       v_ = tf.matmul(u_hat, tf.transpose(w))
       v_hat = tf.nn.l2_normalize(v_)

       u_ = tf.matmul(v_hat, w)
       u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
       w_norm = w / sigma
       w_norm = tf.reshape(w_norm, w_shape)


    return w_norm


def instance_normalization(x, epsilon=1e-05):
    """"Applies Instance Normalization"""
    return tfa.layers.InstanceNormalization(epsilon=epsilon, center=True, scale=True)(x)
