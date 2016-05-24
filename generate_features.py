import os
from numpy import nditer
from numpy import array as nparray
from nibabel import load as load_nii
from scipy.ndimage import label
from scipy.ndimage import center_of_mass
from optparse import OptionParser

def main():
    # parse command line options
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option('-f', '--folder', dest='folder',
                      default='/home/mariano/DATA/CNN-test/Training/',
                      help="read data from FOLDER")
    parser.add_option('-v', '--verbose',
                      action='store_true', dest='verbose',default=False)
    parser.add_option('-p', '--patch-size',
                      action='store', type='int', nargs=3,
                      dest='patch_size', default=(13, 13, 3))
    (options, args) = parser.parse_args()
    
    patients = sorted(os.listdir(options.folder))
    for patient in patients:
        patient_folder = '%s%s/' % (options.folder, patient)
        print 'Executing patient %s' % patient
        mask_nii = load_nii(patient_folder + 'gt_mask.nii')
        mask_img = mask_nii.get_data()
        lesion_centers = get_lesion_centers(mask_img)
        


def get_patches_from_name(filename, centers, patch_size):
    image = load_nii(patient_folder + 'gt_mask.nii')
    patches = get_patches(image, lesion_centers, options.patch_size)
    return patches


def get_patches(image, centers, patch_size):
    patches = []
    list_of_tuples = all(isinstance(centers,tuple))
    sizes_match = [len(center)==len(patch_size) for center in centers]
    if list_of_tuples and sizes_match:
        patch_half = tuple([idx/2 for idx in options.patch_size])
        slices = [[slice(c_idx-p_idx,c_idx+p_idx+1) for (c_idx,p_idx) in zip(center,patch_half)] for center in centers]
        patches = [image[idx] for idx in slices]
    return patches
    
    
def get_lesion_centers(mask):
    labels, nlabels = label(mask)
    all_labels = range(1,nlabels+1)
    centers = [tuple(map(int_round,center)) for center in center_of_mass(mask, labels, all_labels)]
    return centers


def int_round(number):
    return int(round(number))


if __name__ == '__main__':
    main()
