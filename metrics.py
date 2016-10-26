import numpy as np


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
    return 100 * true_positive_seg(target, estimated) / np.sum(as_logical(target))


def fp_fraction_seg(target, estimated):
    return 100.0 * false_positive_seg(target, estimated) / np.sum(as_logical(estimated))


def dsc_seg(target, estimated):
    return 2.0 * true_positive_seg(target, estimated) / np.sum(np.sum(as_logical(target)) + np.sum(as_logical(estimated)))
