import os
import sys
import argparse
from nibabel import load as load_nii
import numpy as np
from skimage.measure import label as bwlabeln
from scipy.ndimage.morphology import binary_erosion as imerode
from sklearn.neighbors import NearestNeighbors
import contextlib


def probabilistic_dsc_seg(target, estimated):
    a = np.array(target).astype(dtype=np.double)
    b = np.array(estimated).astype(dtype=np.double)
    return 2 * np.sum(a * b) / np.sum(np.sum(a) + np.sum(b))


def as_logical(mask):
    return np.array(mask).astype(dtype=np.bool)


def num_regions(mask):
    return np.max(bwlabeln(as_logical(mask)))


def num_voxels(mask):
    return np.sum(as_logical(mask))


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
    b = num_voxels(estimated)
    fpf = 100.0 * false_positive_seg(target, estimated) / b if b > 0 else 0.0
    return fpf


def fp_fraction_det(target, estimated):
    b = np.max(bwlabeln(as_logical(estimated)))
    return 100.0 * false_positive_det(target, estimated) / b if b > 0 else 0.0


def dsc_seg(target, estimated):
    a_plus_b = np.sum(np.sum(as_logical(target)) + np.sum(as_logical(estimated)))
    return 2.0 * true_positive_seg(target, estimated) / a_plus_b if a_plus_b > 0 else 0.0


def dsc_det(target, estimated):
    a_plus_b = (np.max(bwlabeln(as_logical(target))) + np.max(bwlabeln(as_logical(estimated))))
    return 2.0 * true_positive_det(target, estimated) / a_plus_b if a_plus_b > 0 else 0.0


def surface_distance(target, estimated, spacing=[1, 1, 3]):
    a = as_logical(target)
    b = as_logical(estimated)
    a_bound = np.stack(np.where(np.logical_and(a, np.logical_not(imerode(a)))), axis=1) * spacing
    b_bound = np.stack(np.where(np.logical_and(b, np.logical_not(imerode(b)))), axis=1) * spacing
    nbrs_a = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(a_bound) if a_bound.size > 0 else None
    nbrs_b = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(b_bound) if b_bound.size > 0 else None
    distances_a, _ = nbrs_a.kneighbors(b_bound) if nbrs_a and b_bound.size > 0 else ([np.inf], None)
    distances_b, _ = nbrs_b.kneighbors(a_bound) if nbrs_b and a_bound.size > 0 else ([np.inf], None)
    return [distances_a, distances_b]


def average_surface_distance(target, estimated, spacing):
    distances = np.concatenate(surface_distance(target, estimated, spacing))
    return np.mean(distances)


def hausdorff_distance(target, estimated, spacing):
    distances = surface_distance(target, estimated, spacing)
    return np.max([np.max(distances[0]), np.max(distances[1])])


def main():
    @contextlib.contextmanager
    def dummy_file():
        yield None

    # Parse command line options
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    group = parser.add_mutually_exclusive_group()

    folder_help = 'Folder with the files to evaluate. Remember to include a init_names.py for the evaluation pairs.'
    files_help = 'Pair of files to be compared. The first is the GT and the second the file you wnat to evaluate.'

    group.add_argument('-f', '--folder', help=folder_help)
    group.add_argument('--files', nargs=2, help=files_help)
    args = parser.parse_args()

    if args.folder:
        folder_name = args.folder
        sys.path = sys.path + [folder_name]
        from init_names import get_names_from_folder
        gt_names, all_names = get_names_from_folder(folder_name)

    elif args.files:
        folder_name = os.getcwd()
        gt_names = [args.files[0]]
        all_names = [[args.files[1]]]

    with open(os.path.join(folder_name, 'results.csv'), 'w') if args.folder else dummy_file() as f:
        for gt_name, names in zip(gt_names, all_names):
            print('\033[32mEvaluating with ground truth \033[32;1m' + gt_name + '\033[0m')

            gt_nii = load_nii(gt_name)
            gt = gt_nii.get_data()
            spacing = dict(gt_nii.header.items())['pixdim'][1:4]

            for name in names:
                name = ''.join(name)
                print('\033[32m-- vs \033[32;1m' + name + '\033[0m')
                lesion = load_nii(name).get_data()
                dist = average_surface_distance(gt, lesion, spacing)
                tpfv = tp_fraction_seg(gt, lesion)
                fpfv = fp_fraction_seg(gt, lesion)
                dscv = dsc_seg(gt, lesion)
                tpfl = tp_fraction_det(gt, lesion)
                fpfl = fp_fraction_det(gt, lesion)
                dscl = dsc_det(gt, lesion)
                tp = true_positive_det(lesion, gt)
                gt_d = num_regions(gt)
                lesion_s = num_voxels(lesion)
                gt_s = num_voxels(gt)
                pdsc = probabilistic_dsc_seg(gt, lesion)
                if f:
                    measures = (gt_name, name, dist, tpfv, fpfv, dscv, tpfl, fpfl, dscl, tp, gt_d, lesion_s, gt_s)
                    f.write('%s;%s;%f;%f;%f;%f;%f;%f;%f;%d;%d;%d;%d\n' % measures)
                else:
                    measures = (dist, tpfv, fpfv, dscv, tpfl, fpfl, dscl, tp, gt_d, lesion_s, gt_s, pdsc)
                    print('SurfDist TPFV FPFV DSCV TPFL FPFL DSCL TPL GTL Voxels GTV PrDSC')
                    print('%f %f %f %f %f %f %f %d %d %d %d %f' % measures)


if __name__ == '__main__':
    main()
