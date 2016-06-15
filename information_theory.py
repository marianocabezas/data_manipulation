from scipy.stats import entropy
import numpy as np
from numpy import histogramdd
from itertools import chain, combinations


def mutual_information(images):
    # Images should be a list of numpy arrays.

    # In order to compute the mutual information, we will need to compute the
    # entropies for each subset of the images as described ion (Bell 2003)
    # Thus, we create an iterable with all the possible combinations.
    # The next operation is used to exclude the empy set.
    nimages = len(images)
    im_idx = range(0, nimages)
    power_range = range(0, nimages+1)
    image_comb_iter = [combinations(im_idx, power) for power in power_range]
    image_comb = chain.from_iterable(image_comb_iter)
    image_comb.next()

    # For convenience, is also better if we vectorise the images and stack
    # them in a single numpy array. This simplifies the process of computing
    # the histogram (joint in multidimensional cases)
    images_vec = [image.reshape(-1) for image in images]
    np_images = np.stack(images_vec, axis=1)
    histograms = [histogramdd(np_images[:, c], bins=128) for c in image_comb]
    histograms_norm = [(h.reshape(-1) / h.astype(np.float32).sum(), len(h.shape)) for h, e in histograms]
    histograms_non0 = [(h[np.nonzero(h)], s) for h, s in histograms_norm]
    entropies = [-((-1) ** ((nimages - s) % 2)) * entropy(h) for h, s in histograms_non0]
    return np.stack(entropies).sum()
