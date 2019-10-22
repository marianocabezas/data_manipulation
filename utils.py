import time
import os
import re
from itertools import product
import numpy as np
from nibabel import load as load_nii
from scipy.ndimage.morphology import binary_dilation as imdilate
from scipy.ndimage.morphology import binary_erosion as imerode


"""
Utility functions
"""


def color_codes():
    """
    Function that returns a custom dictionary with ASCII codes related to
    colors.
    :return: Custom dictionary with ASCII codes for terminal colors.
    """
    codes = {
        'nc': '\033[0m',
        'b': '\033[1m',
        'k': '\033[0m',
        '0.25': '\033[30m',
        'dgy': '\033[30m',
        'r': '\033[31m',
        'g': '\033[32m',
        'gc': '\033[32m;0m',
        'bg': '\033[32;1m',
        'y': '\033[33m',
        'c': '\033[36m',
        '0.75': '\033[37m',
        'lgy': '\033[37m',
    }
    return codes


def slicing(center_list, size):
    """

    :param center_list:
    :param size:
    :return:
    """
    half_size = tuple(map(lambda ps: ps/2, size))
    ranges = [
        [
            range(
                np.max([c_idx - p_idx, 0]), c_idx + (s_idx - p_idx)
            ) for c_idx, p_idx, s_idx in zip(center, half_size, size)
        ] for center in center_list
    ]
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
    result = list(filter(
        lambda x: not os.path.isdir(x) and re.search(name, x),
        os.listdir(dirname)
    ))

    return os.path.join(dirname, result[0]) if result else None


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


def time_to_string(time_val):
    """
    Function to convert from a time number to a printable string that
     represents time in hours minutes and seconds.
    :param time_val: Time value in seconds (using functions from the time
     package)
    :return: String with a human format for time
    """

    if time_val < 60:
        time_s = '%ds' % time_val
    elif time_val < 3600:
        time_s = '%dm %ds' % (time_val // 60, time_val % 60)
    else:
        time_s = '%dh %dm %ds' % (
            time_val // 3600,
            (time_val % 3600) // 60,
            time_val % 60
        )
    return time_s


def get_mask(mask_name, dilate=0, dtype=np.uint8):
    """
    Function to load a mask image
    :param mask_name: Path to the mask image file
    :param dilate: Dilation radius
    :param dtype: Data type for the final mask
    :return:
    """
    # Lesion mask
    mask_image = load_nii(mask_name).get_data().astype(dtype)
    if dilate > 0:
        mask_d = imdilate(
            mask_image,
            iterations=dilate
        )
        mask_e = imerode(
            mask_image,
            iterations=dilate
        )
        mask_image = np.logical_and(mask_d, np.logical_not(mask_e)).astype(dtype)

    return mask_image


def get_normalised_image(image_name, mask, dtype=np.float32, masked=False):
    """
    Function to a load an image and normalised it (0 mean / 1 standard deviation)
    :param image_name: Path to the image to be noramlised
    :param mask: Mask defining the region of interest
    :param dtype: Data type for the final image
    :param masked: Whether to mask the image or not
    :return:
    """
    mask_bin = mask.astype(np.bool)
    image = load_nii(image_name).get_data().astype(dtype)
    image_mu = np.mean(image[mask_bin])
    image_sigma = np.std(image[mask_bin])
    norm_image = (image - image_mu) / image_sigma

    if masked:
        output = norm_image * mask_bin.astype(dtype)
    else:
        output = norm_image

    return output
