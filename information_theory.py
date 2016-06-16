from scipy.stats import entropy
import numpy as np
from numpy import histogramdd, histogram
from itertools import chain, combinations


def mutual_information(images, bins=256):
    # Images should be a list of numpy arrays.

    # In order to compute the mutual information, we will need to compute the
    # entropies for each subset of the images as described ion (Bell 2003)
    # Thus, we create an iterable with all the possible combinations.
    nimages = len(images)
    im_idx = range(0, nimages)
    power_range = range(1, nimages+1)
    image_comb_iter = [combinations(im_idx, power) for power in power_range]
    image_comb = chain.from_iterable(image_comb_iter)

    # For convenience, is also better if we vectorise the images and stack
    # them in a single numpy array. This simplifies the process of computing
    # the histogram (joint in multidimensional cases)
    images_vec = [image.reshape(-1) for image in images]
    np_images = np.stack(images_vec, axis=1)
    histograms = [histogramdd(np_images[:, c], bins=bins) for c in image_comb]
    histograms_norm = [(h.reshape(-1) / h.astype(np.float32).sum(), len(h.shape)) for h, e in histograms]
    histograms_non0 = [(h[np.nonzero(h)], s) for h, s in histograms_norm]
    informations = [-((-1) ** ((nimages - s) % 2)) * entropy(h) for h, s in histograms_non0]
    return np.stack(informations).sum()


def entropies(images, bins=256):
    # Images should be a list of numpy arrays.
    histograms = [histogram(image.reshape(-1), bins=bins) for image in images]
    return [entropy(h[np.nonzero(h)] / h.sum()) for h, s in histograms]


def joint_entropy(images, bins=256):
    # Images should be a list of numpy arrays.
    np_images = np.stack([image.reshape(-1) for image in images], axis=1)
    h, s = histogramdd(np_images, bins=bins)
    return entropy(h[np.nonzero(h)] / h.sum())
