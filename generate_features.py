#!/usr/bin/env python2.7
import os
# from numpy import nditer
# from numpy import array as nparray
from nibabel import load as load_nii
from scipy.ndimage import label
from scipy.ndimage import center_of_mass
from optparse import OptionParser
import numpy as np
from operator import add

def main():
    # Parse command line options
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option('-f', '--folder', dest='folder',
                      default='/home/mariano/DATA/Challenge/',
                      help="read data from FOLDER")
    parser.add_option('-v', '--verbose',
                      action='store_true', dest='verbose', default=False)
    parser.add_option('-p', '--patch-size',
                      action='store', type='int', nargs=3,
                      dest='patch_size', default=(15, 15, 15))
    parser.add_option('--use-gado',
                      action='store_true', dest='use_gado')
    parser.add_option('--no-gado',
                      action='store_false', dest='use_gado', default=False)
    parser.add_option('--gado',
                      action='store', dest='gado', type='string', default='GADO_preprocessed.nii.gz')
    parser.add_option('--use-flair',
                      action='store_true', dest='use_flair')
    parser.add_option('--no-flair',
                      action='store_false', dest='use_flair', default=True)
    parser.add_option('--flair',
                      action='store', dest='flair', type='string', default='FLAIR_preprocessed.nii.gz')
    parser.add_option('--use-pd',
                      action='store_true', dest='use_pd')
    parser.add_option('--no-pd',
                      action='store_false', dest='use_pd', default=True)
    parser.add_option('--pd',
                      action='store', dest='pd', type='string', default='DP_preprocessed.nii.gz')
    parser.add_option('--use-t2',
                      action='store_true', dest='use_t2')
    parser.add_option('--no-t2',
                      action='store_false', dest='use_t2', default=True)
    parser.add_option('--t2',
                      action='store', dest='t2', type='string', default='T2_preprocessed.nii.gz')
    parser.add_option('--use-t1',
                      action='store_true', dest='use_t1')
    parser.add_option('--no-t1',
                      action='store_false', dest='use_t1', default=True)
    parser.add_option('--t1',
                      action='store', dest='t1', type='string', default='T1_preprocessed.nii.gz')
    parser.add_option('--mask',
                      action='store', dest='mask', type='string', default='Consensus.nii.gz')
    (options, args) = parser.parse_args()

    files = sorted(os.listdir(options.folder))
    patients = [f for f in files if os.path.isdir(os.path.join(options.folder, f))]
    for patient in patients:
        patient_folder = os.path.join(options.folder, patient)
        print 'Executing patient %s' % patient

        mask_nii = load_nii(os.path.join(patient_folder, options.mask))
        mask_img = mask_nii.get_data()
        lesion_centers = get_mask_voxels(mask_img)

        flair = None
        pd = None
        t1 = None
        t2 = None
        gado = None

        if options.use_flair:
            flair = get_patches_from_name(os.path.join(patient_folder, options.flair),
                                          lesion_centers,
                                          options.patch_size
                                          )

        if options.use_pd:
            pd = get_patches_from_name(os.path.join(patient_folder, options.pd),
                                       lesion_centers,
                                       options.patch_size
                                       )

        if options.use_t1:
            t1 = get_patches_from_name(os.path.join(patient_folder, options.t1),
                                       lesion_centers,
                                       options.patch_size
                                       )

        if options.use_t2:
            t2 = get_patches_from_name(os.path.join(patient_folder, options.t2),
                                       lesion_centers,
                                       options.patch_size
                                       )

        if options.use_gado:
            gado = get_patches_from_name(os.path.join(patient_folder, options.flair),
                                         lesion_centers,
                                         options.patch_size
                                         )

        patches = np.stack([np.array(data) for data in [flair, pd, t2, gado, t1] if data is not None], axis=1)

        print 'Our final vector\'s size = (' + ','.join([str(num) for num in patches.shape]) + ')'


def get_patches_from_name(filename, centers, patch_size):
    image = load_nii(filename).get_data()
    patches = get_patches(image, centers, patch_size)
    return patches


def get_patches(image, centers, patch_size):
    patches = []
    list_of_tuples = all([isinstance(center, tuple) for center in centers])
    sizes_match = [len(center) == len(patch_size) for center in centers]
    if list_of_tuples and sizes_match:
        patch_half = tuple([idx/2 for idx in patch_size])
        new_centers = [map(add, center, patch_half) for center in centers]
        padding = tuple((idx, idx) for idx in patch_half)
        new_image = np.pad(image, padding, mode='constant', constant_values=0)
        slices = [
            [slice(c_idx-p_idx, c_idx+p_idx+1) for (c_idx, p_idx) in zip(center, patch_half)]
            for center in new_centers
        ]
        patches = [new_image[idx] for idx in slices]
    return patches
    
    
def get_mask_voxels(mask):
    indices = np.stack(np.nonzero(mask), axis=1)
    indices = [tuple(idx) for idx in indices]
    return indices


def get_mask_centers(mask):
    labels, nlabels = label(mask)
    all_labels = range(1, nlabels+1)
    centers = [tuple(map(int_round, center)) for center in center_of_mass(mask, labels, all_labels)]
    return centers


def int_round(number):
    return int(round(number))


if __name__ == '__main__':
    main()
