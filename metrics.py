import os
import sys
import argparse
from nibabel import load as load_nii
import numpy as np
from skimage.measure import label as bwlabeln
from scipy.ndimage.morphology import binary_erosion as imerode
from sklearn.neighbors import NearestNeighbors
import contextlib


def regionprops(mask):
    blobs = bwlabeln(as_logical(mask))
    labels = filter(bool, np.unique(blobs))
    areas = [np.count_nonzero(blobs == l) for l in labels]
    return blobs, labels, areas


def masks_by_size(mask, sizes):
    blobs, labels, areas = regionprops(mask)
    labels_list = [[l for l, a in zip(labels, areas) if a >= mins & a < maxs]
                   for mins, maxs in zip(sizes[:-1], sizes[1:])]
    labels_list.append([l for l, a in zip(labels, areas) if a > sizes[-1]])
    submasks = [reduce(lambda x, y: np.logical_or(x, y), [np.equal(blobs, l) for l in nu_labels if nu_labels])
                if nu_labels else np.zeros_like(mask) for nu_labels in labels_list]
    return submasks


def analysis_by_sizes(target, estimated, sizes):
    a = as_logical(target)
    b = as_logical(estimated)
    a_sub = masks_by_size(a, sizes)
    fp_sub = masks_by_size(np.logical_and(np.logical_not(a), b), sizes)
    tpd_list = [true_positive_det(a_i, b) for a_i in a_sub]
    tps_list = [true_positive_seg(a_i, b) for a_i in a_sub]
    fpd_list = [len(filter(bool, np.unique(bwlabeln(fp_i)))) for fp_i in fp_sub]
    fps_list = [np.count_nonzero(fp_i) for fp_i in fp_sub]
    gtd_list = [len(filter(bool, np.unique(bwlabeln(a_i)))) for a_i in a_sub]
    gts_list = [np.count_nonzero(a_i) for a_i in a_sub]
    tpf_list = [tp_i/gt_i for tp_i, gt_i in zip(tpd_list, gtd_list)]
    fpf_list = [fp_i / (fp_i+tp_i) for tp_i, fp_i in zip(tpd_list, fpd_list)]
    dscd_list = [2 * tp_i / (tp_i + fp_i + gt_i) for tp_i, fp_i, gt_i in zip(tpd_list, fpd_list, gtd_list)]
    dscs_list = [2 * tp_i / (tp_i + fp_i + gt_i) for tp_i, fp_i, gt_i in zip(tps_list, fps_list, gts_list)]
    return tpf_list, fpf_list, dscd_list, dscs_list


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
    return np.count_nonzero(np.logical_and(a, b))


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
    return np.count_nonzero(np.logical_and(a, b))


def false_positive_seg(target, estimated):
    a = as_logical(target)
    b = as_logical(estimated)
    return np.count_nonzero(np.logical_and(np.logical_not(a), b))


def false_negative_seg(target, estimated):
    a = as_logical(target)
    b = as_logical(estimated)
    return np.count_nonzero(np.logical_and(a, np.logical_not(b)))


def tp_fraction_seg(target, estimated):
    return 100.0 * true_positive_seg(target, estimated) / np.count_nonzero(as_logical(target)) \
        if np.count_nonzero(as_logical(target)) > 0 else 0


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
    a_plus_b = np.sum(np.sum(as_logical(target)) + np.count_nonzero(as_logical(estimated)))
    return 2.0 * true_positive_seg(target, estimated) / a_plus_b if a_plus_b > 0 else 0.0


def dsc_det(target, estimated):
    a_plus_b = (np.max(bwlabeln(as_logical(target))) + np.max(bwlabeln(as_logical(estimated))))
    return 2.0 * true_positive_det(target, estimated) / a_plus_b if a_plus_b > 0 else 0.0


def eucl_distance(a, b):
    nbrs_a = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(a) if a.size > 0 else None
    nbrs_b = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(b) if b.size > 0 else None
    distances_a, _ = nbrs_a.kneighbors(b) if nbrs_a and b.size > 0 else ([np.inf], None)
    distances_b, _ = nbrs_b.kneighbors(a) if nbrs_b and a.size > 0 else ([np.inf], None)

    return [distances_a, distances_b]


def surface_distance(target, estimated, spacing=list((1, 1, 3))):
    a = as_logical(target)
    b = as_logical(estimated)
    a_bound = np.stack(np.where(np.logical_and(a, np.logical_not(imerode(a)))), axis=1) * spacing
    b_bound = np.stack(np.where(np.logical_and(b, np.logical_not(imerode(b)))), axis=1) * spacing
    return eucl_distance(a_bound, b_bound)


def mask_distance(target, estimated, spacing=list((1, 1, 3))):
    a = as_logical(target)
    b = as_logical(estimated)
    a_full = np.stack(np.where(a), axis=1) * spacing
    b_full = np.stack(np.where(b), axis=1) * spacing
    return eucl_distance(a_full, b_full)


def average_surface_distance(target, estimated, spacing):
    distances = np.concatenate(surface_distance(target, estimated, spacing))
    return np.mean(distances)


def hausdorff_distance(target, estimated, spacing):
    distances = surface_distance(target, estimated, spacing)
    return np.max([np.max(distances[0]), np.max(distances[1])])


def modified_hausdorff_distance(target, estimated, spacing):
    distances = mask_distance(target, estimated, spacing)
    return np.max([np.mean(distances[0]), np.mean(distances[1])])


def main():
    @contextlib.contextmanager
    def dummy_file():
        yield None

    # Parse command line options
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    group_in = parser.add_mutually_exclusive_group()
    group_out = parser.add_mutually_exclusive_group()

    folder_help = 'Folder with the files to evaluate. Remember to include a init_names.py for the evaluation pairs.'
    files_help = 'Pair of files to be compared. The first is the GT and the second the file you wnat to evaluate.'

    group_in.add_argument('-f', '--folder', help=folder_help)
    group_in.add_argument('--files', nargs=2, help=files_help)

    general_help = 'General evaluation. Based on the 2008 MS challenge''s measures. ' \
                   'Volume and detection absolute measures are also included for all the images.'
    sizes_help = 'Evaluation based on region sizes. ' \
                 'Includes TPF, FPF and DSC for detection and DSC for segmentation. ' \
                 'The size of TP is determined by the GT size, while the FP size is determined by the FP lesion.'

    group_out.add_argument('-g', '--general', help=general_help)
    group_out.add_argument('-s', '--sizes', dest='sizes', nargs='+', type=int, default=[3, 11, 51], help=sizes_help)

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

                if args.general:
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
                elif args.sizes:
                    sizes = args.sizes
                    tpf, fpf, dscd, dscs = analysis_by_sizes(gt, lesion, sizes)
                    names = '%s;%s;' % (gt_name, name)
                    measures = ';'.join(['%f;%f;%f;%f' % (tpf_i, fpf_i, dscd_i, dscs_i)
                                         for tpf_i, fpf_i, dscd_i, dscs_i in zip(tpf, fpf, dscd, dscs)])
                    if f:
                        f.write(names + measures + '\n')
                    else:
                        intervals = ['\t\t[%d-%d)\t\t|' % (mins, maxs) for mins, maxs in zip(sizes[:-1], sizes[1:])]
                        intervals = ''.join(intervals) + '\t\t[%d-inf)\t|' % sizes[-1]
                        measures_s = 'TPF\tFPF\tDSCd\tDSCs\t|' * len(sizes)
                        measures = ''.join(['%.2f\t%.2f\t%.2f\t%.2f\t|' % (tpf_i, fpf_i, dscd_i, dscs_i)
                                            for tpf_i, fpf_i, dscd_i, dscs_i in zip(tpf, fpf, dscd, dscs)])
                        print(intervals)
                        print(measures_s)
                        print(measures)


if __name__ == '__main__':
    main()
