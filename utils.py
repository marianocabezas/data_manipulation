from __future__ import print_function
import time
import os
import re
from itertools import product
import numpy as np


def slicing(center_list, size):
    """

    :param center_list:
    :param size:
    :return:
    """
    half_size = tuple(map(lambda ps: ps/2, size))
    ranges = map(
        lambda center: map(
            lambda (c_idx, p_idx, s_idx): range(
                np.max([c_idx - p_idx, 0]), c_idx + (s_idx - p_idx)
            ),
            zip(center, half_size, size)),
        center_list
    )
    slices = np.concatenate(
        map(
            lambda x: np.stack(list(product(*x)), axis=1),
            ranges
        ),
        axis=1
    )
    return slices


def find_file(name, dirname):
    """

    :param name:
    :param dirname:
    :return:
    """
    result = filter(
        lambda x: not os.path.isdir(x) and re.search(name, x),
        os.listdir(dirname)
    )

    return result[0] if result else None


def print_message(message):
    """
    Function to print a message with a custom specification
    :param message: Message to be printed.
    :return: None.
    """
    c = color_codes()
    dashes = ''.join(['-'] * (len(message) + 11))
    print(dashes)
    print(
        '%s[%s]%s %s' %
        (c['c'], time.strftime("%H:%M:%S", time.localtime()), c['nc'], message)
    )
    print(dashes)
