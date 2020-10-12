"""
    Assign a different colour to each type of message.

    Author: Mar Alguacil
"""
from termcolor import colored
import tensorflow as tf

def INFO(*msg):
    if (len(msg) == 1):
        print(colored(*msg, 'cyan'))
    else:
        print(colored(msg, 'cyan'))

def INFO_COLOR(msg):
    return colored(msg, 'cyan')


def WARN(*msg):
    if (len(msg) == 1):
        print(colored(*msg, 'yellow'))
    else:
        print(colored(msg, 'yellow'))

def WARN_COLOR(msg):
    return colored(msg, 'yellow')


def ERROR(*msg):
    if (len(msg) == 1):
        print(colored(*msg, 'red'))
    else:
        print(colored(msg, 'red'))

def ERROR_COLOR(msg):
    return colored(msg, 'red')


def tf_print_float(number, digits=3):
    number = reduce_precision(number, digits)

    if number>=0:
        tf.print(' ', end='', sep='')

    # if tf.round(number) in range(1,10):
    if number>3 and number<10:
        tf.print(' ', end='', sep='')

    tf.print(number, end='', sep='')

    n = number*tf.pow(10.0, digits)
    if n%tf.pow(10.0, digits) == 0:
        tf.print('.', '0'*digits, end='', sep='')
    else:
        while n!=number and n%10==0 :
            tf.print(0, end='', sep='')
            n/=10

def reduce_precision(number, precision=0):
    """Reduces precision floats."""
    # return tf.math.round(number * tf.pow(10.0, digits)) * tf.pow(10.0, -digits)
    N = tf.pow(2.0, precision)
    return tf.round(N * number)/N
