import numpy as np
from skimage.measure import label as bwlabeln
from scipy.ndimage.morphology import binary_erosion as imerode
from sklearn.neighbors import NearestNeighbors


def probabilistic_dsc_seg(target, estimated):
    a = np.array(target).astype(dtype=np.double)
    b = np.array(estimated).astype(dtype=np.double)
    return 2 * np.sum(a * b) / np.sum(np.sum(a) + np.sum(b))


def as_logical(mask):
    return np.array(mask).astype(dtype=np.bool)


def true_positive_seg(target, estimated):
    a = as_logical(target)
    b = as_logical(estimated)
    return np.sum(np.logical_and(a, b))


def true_positive_det(target, estimated):
    a = bwlabeln(as_logical(target))
    b = as_logical(estimated)
    return np.min([np.sum([np.logical_and(b, a == (i+1)).any() for i in range(np.max(a))]), np.max(bwlabeln(b))])


def false_positive_det(target, estimated):
    a = as_logical(target)
    b = bwlabeln(as_logical(estimated))
    tp_labels = np.unique(a * b)
    fp_labels = np.unique(np.logical_not(a) * b)
    return len([label for label in fp_labels if label not in tp_labels])


def true_negative_seg(target, estimated):
    a = as_logical(target)
    b = as_logical(estimated)
    return np.sum(np.logical_and(a, b))


def false_positive_seg(target, estimated):
    a = as_logical(target)
    b = as_logical(estimated)
    return np.sum(np.logical_and(np.logical_not(a), b))


def false_negative_seg(target, estimated):
    a = as_logical(target)
    b = as_logical(estimated)
    return np.sum(np.logical_and(a, np.logical_not(b)))


def tp_fraction_seg(target, estimated):
    return 100.0 * true_positive_seg(target, estimated) / np.sum(as_logical(target)) \
        if np.sum(as_logical(target)) > 0 else 0


def tp_fraction_det(target, estimated):
    return 100.0 * true_positive_det(target, estimated) / np.max(bwlabeln(as_logical(target)))


def fp_fraction_seg(target, estimated):
    return 100.0 * false_positive_seg(target, estimated) / np.sum(as_logical(estimated))


def fp_fraction_det(target, estimated):
    return 100.0 * false_positive_det(target, estimated) / np.max(bwlabeln(as_logical(estimated)))


def dsc_seg(target, estimated):
    a_plus_b = np.sum(np.sum(as_logical(target)) + np.sum(as_logical(estimated)))
    return 2.0 * true_positive_seg(target, estimated) / a_plus_b


def dsc_det(target, estimated):
    a_plus_b = (np.max(bwlabeln(as_logical(target))) + np.max(bwlabeln(as_logical(estimated))))
    return 2.0 * true_positive_det(target, estimated) / a_plus_b


def average_surface_distance(target, estimated, spacing=[1, 1, 3]):
    a = as_logical(target)
    b = as_logical(estimated)
    a_bound = np.stack(np.where(np.logical_and(a, np.logical_not(imerode(a)))), axis=1) * spacing
    b_bound = np.stack(np.where(np.logical_and(b, np.logical_not(imerode(b)))), axis=1) * spacing
    nbrs_a = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(a_bound)
    nbrs_b = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(b_bound)
    distances_a, _ = nbrs_a.kneighbors(b_bound)
    distances_b, _ = nbrs_b.kneighbors(a_bound)
    distances = np.concatenate([distances_a, distances_b])
    return np.mean(distances)

