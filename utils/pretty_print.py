"""
    Assign a different colour to each type of message.

    Author: Mar Alguacil
"""
from termcolor import colored

def INFO(*msg):
    if (len(msg) == 1):
        print(colored(*msg, 'cyan'))
    else:
        print(colored(msg, 'cyan'))

def WARN(*msg):
    if (len(msg) == 1):
        print(colored(*msg, 'yellow'))
    else:
        print(colored(msg, 'yellow'))

def ERROR(*msg):
    if (len(msg) == 1):
        print(colored(*msg, 'red'))
    else:
        print(colored(msg, 'red'))

def ERROR_COLOR(msg):
    return colored(msg, 'red')
