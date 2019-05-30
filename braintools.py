from __future__ import print_function
import os
from nibabel import load as load_nii
from utils import print_message
import sitk


def atlas_registration(
        atlas,
        reference,
        atlases_pr,
        structures=None,
        path=None,
        verbose=1,

):
    """
    Function to perform atlas registration using Demons from SimpleITK.
    :param atlas: Path to the template image file for the atlas.
    :param reference: Path to the refence image file for the atlas.
    :param atlases_pr: List of paths to the probabilistic atlas files.
    :param structures: Path to a structural mask for the atlas.
    :param path: Path where the output will be saved.
    :param verbose: Verbose levels for this tool. The minimum value must be 1.
     For this level of verbosity, only "required" messages involving each step
     and likelihood will be shown. For the next level, various debugging
     options related to the expectation maximisation will be shown.
    """
    # > Atlas registration
    # Affine registration for Demons.
    # if not find_file('atlas_affine.nii.gz', temp_path):
    if verbose > 0:
        print_message(
            'Atlas registration - %s (%s)' % (timepoint, patient)
        )
    if verbose > 1:
        print_message(
            '- Affine registration - %s (%s)' % (timepoint, patient)
        )
    affine = sitk.itkaffine(t1_name, t1_atlas)
    sitk.itkresample(
        reference, atlas, affine,
        path=path, name='atlas_affine'
    )
    if structures is not None:
        sitk.itkresample(
            reference, structures, affine, interpolation='nn',
            path=path, name='structures_affine'
        )
    atlases_affine = map('atlas_affine_pr%d.nii.gz')
    map(
        lambda (pr_i, name): sitk.itkresample(
            reference, ventricles_pr, affine,
            path=path, name=name
        ),
        zip(atlases_pr, atlases_affine)
    )

    # Histogram matching
    atlas_affine = os.path.join(path, 'atlas_affine.nii.gz')
    if verbose > 1:
        print_message(
            '- Histogram matching - %s (%s)' % (timepoint, patient)
        )
    sitk.itkhist_match(
        t1_name, atlas_affine, match_points=24,
        path=path, name='atlas'
    )

    # Demons computation
    if verbose > 1:
        print_message(
            '- Demons registration - %s (%s)' % (timepoint, patient)
        )
    atlas_matched = os.path.join(
        path, 'atlas_corrected_matched.nii.gz'
    )
    mask = os.path.join(processed_path, 'union_brainmask.nii.gz')
    sitk.itkdemons(
        reference, atlas_matched, mask, path=path, name='atlas',
        steps=100, sigma=1.5
    )

    df = os.path.join(temp_path, 'atlas_multidemons_deformation.nii.gz')
    if structures is not None:
        sitk.itkwarp(
            reference, os.path.join(temp_path, 'structures_affine.nii.gz'),
            df, interpolation='nn', path=path, name='atlas_ventricles'
        )
    map(
        lambda (i, pr_i): sitk.itkwarp(
            reference, atlases_affine, df,
            path=path, name='atlas_pr%d' % i
        ),
        enumerate(atlases_pr)
    )
    sitk.itkwarp(
        reference, atlas_matched, df,
        path=path, name='atlas_demons'
    )

    # > Similarity computing
    # First we pad the images to extract the mask patches
    if verbose > 0:
        print_message(
            'Atlas similarity - %s (%s)' % (timepoint, patient)
        )
    similarity_name = os.path.join(path, 'atlas_similarity.nii.gz')
    if find_file('atlas_similarity.nii.gz', path) is None:
        t1nii = load_nii(reference)
        t1 = t1nii.get_data()
        atlas_demons = load_nii(
            os.path.join(path, 'atlas_demons.nii.gz')
        ).get_data()
        padding = tuple(
            (idx, size - idx) for idx, size in zip(patch_half, patch_size)
        )
        f_pad = np.pad(t1, padding, 'constant', constant_values=0.0)
        m_pad = np.pad(atlas_demons, padding, 'constant', constant_values=0.0)

        # Then we compute the centers (slices) for the mask voxels
        new_centers = map(
            lambda center: map(add, center, patch_half),
            centers
        )
        [slices_x, slices_y, slices_z] = slicing(new_centers, patch_size)

        # Finally, we just compute cross-correlation on the patches and
        # save the information inside the voxel
        similarity = np.zeros_like(t1)
        f_pad_s = np.stack(
            np.split(f_pad[slices_x, slices_y, slices_z], len(centers)),
            axis=1
        )
        m_pad_s = np.stack(
            np.split(m_pad[slices_x, slices_y, slices_z], len(centers)),
            axis=1
        )
        # The variable is called cross variance, because it would be the
        # variance if the patches were the same.
        f_var = f_pad_s - f_pad_s.mean(axis=0)
        m_var = m_pad_s - m_pad_s.mean(axis=0)
        slices_xvar = np.mean(f_var * m_var, axis=0)
        slices_std = f_pad_s.std(axis=0) * m_pad_s.std(axis=0)
        slices_std[slices_std <= 0] = np.finfo(float).eps
        slices_xcorr = np.fabs(slices_xvar / slices_std)
        slices_xcorr[slices_xcorr > 1] = 1
        similarity[x, y, z] = slices_xcorr
        t1nii.get_data()[:] = similarity
        t1nii.to_filename(similarity_name)

        if verbose > 1:
            print_message(
                '- Similarity range = [%f, %f]' % (
                    similarity.min(), similarity.max()
                )
            )


def tissue_pve(
        images,
        mask,
        similarity,
        atlases_pr,
        path=None,
        patch_size=(3, 3, 3),
        th=0.75,
        max_iter=10,
        alpha=2.0,
        pv_classes=list([(1, 2)]),
        verbose=1,
):
    """
    Function that performs atlas registration and tissue segmentation using a
     probabilistic atlas that differentiates between cortical CSF and the
     ventricles.
    :param images: List of paths to the images to be used for segmentation.
    :param mask: Path to the mask of the brain.
    :param similarity: Path to the atlas similarity image.
    :param atlases_pr: Probabilistic atlases.
    :param path: Path where the output will be saved.
    :param patch_size: Patch size used to compute neighbour probabilities and
     the similarity image.
    :param th: Threshold for the tissue segmentation step. This threshold is
     used for the trimmed likelihood estimator during the expectation
     maximisation approach. Unlike previous C++ versions of this method, this
     threshold is adaptive to allow the mean and covariance computation even
     with values under this threshold for each class.
    :param max_iter: Max number of iterations for the expectation maximisation
     approach.
    :param alpha: Parameter for the threshold estimation during the "lesion
     segmentation" thresholding.
    :param pv_classes: List of tuple pairs of values that represent the tissue
     classes. 0: Ventricles-CSF, 1: External-CSF, 2: GM, 3: WM.
    :param verbose: Verbose levels for this tool. The minimum value must be 1.
     For this level of verbosity, only "required" messages involving each step
     and likelihood will be shown. For the next level, various debugging
     options related to the expectation maximisation will be shown.
    :return: None.
    """

    # Init
    if verbose > 0:
        print_message(
            'Tissue segmentation - %s (%s)' % (timepoint, patient)
        )

    patch_half = tuple(map(lambda ps: ps/2, patch_size))

    prnii = load_nii(atlases_pr[0])
    masknii = load_nii(mask)
    mask_im = mask_ni.get_data()
    flair = load_nii(images[-1]).get_data()

    '''Tissue segmentation'''
    if verbose > 0:
        print_message(
            '- Segmentation start - %s (%s)' % (timepoint, patient)
        )
    if find_file('brain.nii.gz', path) is None:
        # Init
        for a in atlases_pr:
            a[a < 0] = 0
        new_centers = map(lambda center: map(add, center, patch_half), centers)
        [slices_x, slices_y, slices_z] = slicing(new_centers, patch_size)

        # Partial volume atlas creation and atlas probability normalisation
        if verbose > 0:
            print_message('- Partial volume class atlas creation')

        # Now we'll create the partial volume atlases. That means that we need
        # to merge the atlases of both classes, and renormalize everything.
        # Remember: The sum of all atlases for  agiven voxel should be 1.
        atlases = atlases_pr + map(
            lambda (i0, i1): atlases_pr[i0] + atlases_pr[i1] / 2.0,
            pv_classes
        )
        if verbose > 1:
            iapr_s = ' '.join(
                map(
                    lambda pr_i: '[%.5f, %.5f]' % (pr_i.min(), pr_i.max()),
                    np_atlases
                )
            )
            print_message('-- initial atlas ranges = %s)' % iapr_s)

        # Here we renormalize the probabilities.
        atlases_sum = np.sum(atlases, axis=0)
        if verbose > 1:
            print_message(
                '-- atlas sum ranges = [%.5f, %.5f]' % (
                    atlases_sum.min(), atlases_sum.max()
                )
            )
        nonzero_sum = np.nonzero(atlases_sum)
        for a in atlases:
            a[nonzero_sum] = a[nonzero_sum] / atlases_sum[nonzero_sum]
        if verbose > 1:
            apr_s = ' '.join(
                map(
                    lambda pr_i: '[%.5f, %.5f]' % (pr_i.min(), pr_i.max()),
                    atlases
                )
            )
            print_message('-- atlas ranges = %s)' % apr_s)

        # First we create the neighbourhood priors as stated on the paper. These are initial priors and
        # they are computed differently for pure and partial volume classes. These values need to be
        # updated at each step.
        if verbose > 0:
            print('- Neighborhood priors (initial)')
        npr = map(np.zeros_like, atlases)
        pure_tissues = len(np_atlases)
        padding = tuple(
            (idx, size - idx) for idx, size in zip(patch_half, patch_size)
        )
        padded_atlases = map(
            lambda a_i: np.pad(a_i, padding, 'constant', constant_values=0.0),
            atlases
        )
        npr_values = map(
            lambda pr_i: np.mean(
                np.stack(
                    np.split(
                        pr_i[slices_x, slices_y, slices_z],
                        len(centers)
                    ),
                    axis=1
                ),
                axis=0
            ),
            padded_atlases
        )
        for npr_i, values in zip(npr, npr_values):
            npr_i[x, y, z] = values

        # Now we create the atlas priors. These are constant and are computed using the atlas priors and the
        # similarity image. Since they are constant, we'll compute them once only. In the C++ code
        # these maps were recomputed at each iteration. We'll just do it once here.
        # Pure tissue classes
        if verbose > 0:
            print_message('- Atlas priors (initial)')
        apr = map(lambda pr_i: pr_i * similarity, atlases)

        # Finally we create the initial posterior probabilities. This are defined by the Gauss
        # distribution probability function. For pure tissues we estimate the mu and sigma from
        # the data and the atlas priors. For the partial volumes we average them.
        # However, for the initial estimate, we'll use the priors and we'll update cpr during
        # the next iterations. For convenience I'm keeping expectation and maximisation as functions
        # for the current function. They are before the loop for better readability.
        # Pure tissue classes
        if verbose > 0:
            print('- Posterior probabilities (initial)')
        ppr = map(np.copy, atlases)
        if verbose > 1:
            ppr_s = ' '.join(
                map(
                    lambda pr_i: '[%.5f, %.5f]' % (pr_i.min(), pr_i.max()),
                    ppr
                )
            )
            print_message('\t  (ppr ranges = %s)' % ppr_s)

        # Initial values for loop
        sum_log_ant = -np.inf

        sum_ppr = np.sum(ppr, axis=0)
        sum_log = np.sum(map(
            lambda pr_i: np.sum(np.log(pr_i[pr_i > 0] / sum_ppr[pr_i > 0])),
            ppr
        ))
        i = 0
        if verbose > 0:
            print_message('- Main EM loop (initial log likelihood = %.2e)' % sum_log)
        while i < max_iter and np.fabs(sum_log_ant - sum_log) > np.finfo(float).eps:
            i += 1
            if verbose > 1:
                print_message('---- Iteration %2d' % i)
            elif verbose > 0:
                print('-- Iteration %2d' % i, end=' ')
                sys.stdout.flush()
            # <Expectation step>
            # The pure parameters are updated from the data, while the partial ones
            # are updated using the pure ones.
            min_pure_ppr = np.min(map(np.max, ppr[:pure_tissues]))
            adaptive_th = min_pure_ppr / 2.0 if min_pure_ppr < th else th
            if verbose > 1:
                print_message('--- expectation')
            elif verbose > 0:
                print('<expectation>', end=' ')
                sys.stdout.flush()
            pure_params = map(
                lambda pr_i: expectation(images, pr_i, adaptive_th, verbose),
                ppr[:pure_tissues]
            )
            pv_params = map(
                lambda (i0, i1): tuple(map(
                    lambda (p0, p1): (p0 + p1) / 2,
                    zip(pure_params[i0], pure_params[i1])
                )),
                pv_classes
            )
            params = pure_params + pv_params
            # <Maximisation step>
            # The conditional probability is computed using the Gaussian mixture model defined
            # previously. Since the mean and covariance matrix are already updated, the conditional
            # probability is computed using the same equation for both pure and partial classes.
            # Conditional probability (Gaussian)
            if verbose > 1:
                print_message('--- maximisation')
            elif verbose > 0:
                print('<maximisation>', end=' ')
                sys.stdout.flush()
            cpr = map(
                lambda (mu_i, sigma_i): maximisation(
                    images,
                    mask_im,
                    mu_i,
                    sigma_i,
                    verbose
                ),
                params
            )
            # Priors: Atlas weighted by similarity + Neighbourhood weighted by inverse similarity
            priors = map(
                lambda (apr_i, pr_i): apr_i + (1 - similarity) * pr_i,
                zip(apr, npr)
            )
            # Posterior probability = cpr * priors
            ppr = map(
                lambda (cpr_i, prior_i): mask_im * cpr_i * prior_i,
                zip(cpr, priors)
            )
            # Posterior are normalised with the sum of the probabilities for each class
            sum_ppr = np.sum(ppr, axis=0)
            nonzero_pr = np.nonzero(sum_ppr > 0)
            for ppr_i in ppr:
                ppr_i[nonzero_pr] = ppr_i[nonzero_pr] / sum_ppr[nonzero_pr]
                ppr_i[ppr_i < 0] = 0
                ppr_i[ppr_i > 1] = 1

            if verbose > 1:
                print_message('-- similarity range = [%.5f, %.5f]' % (
                    similarity.min(), similarity.max()
                ))
                npr_s = ' '.join(
                    map(
                        lambda pr_i: '[%.5f, %.5f]' % (pr_i.min(), pr_i.max()),
                        npr
                    )
                )
                print_message('--  (npr ranges = %s)' % npr_s)
                apr_s = ' '.join(
                    map(
                        lambda pr_i: '[%.5f, %.5f]' % (pr_i.min(), pr_i.max()),
                        apr
                    )
                )
                print_message('--  (apr ranges = %s)' % apr_s)
                prpr_s = ' '.join(
                    map(
                        lambda pr_i: '[%.5f, %.5f]' % (pr_i.min(), pr_i.max()),
                        priors
                    )
                )
                print_message('--  (priors ranges = %s)' % prpr_s)
                cpr_s = ' '.join(
                    map(
                        lambda pr_i: '[%.2e, %.2e]' % (pr_i.min(), pr_i.max()),
                        cpr
                    )
                )
                print_message('--  (conditional probability ranges = %s)' % cpr_s)
                ppr_s = ' '.join(
                    map(
                        lambda pr_i: '[%f, %f]' % (pr_i.min(), pr_i.max()),
                        ppr
                    )
                )
                print('\t  (posterior probability ranges = %s)' % ppr_s)

            # We prepare the data for the next iteration
            npr_values = map(
                lambda pr_i: np.mean(
                    np.stack(
                        np.split(
                            pr_i[slices_x, slices_y, slices_z],
                            len(centers)
                        ),
                        axis=1
                    ),
                    axis=0
                ),
                map(
                    lambda a_i: np.pad(
                        a_i,
                        padding,
                        'constant',
                        constant_values=0.0
                    ),
                    ppr
                )
            )
            for npr_i, values in zip(npr, npr_values):
                npr_i[x, y, z] = values

            # Update the objective function
            sum_log_ant = sum_log
            sum_log = np.sum(map(lambda pr_i: np.sum(np.log(pr_i[pr_i > 0])), ppr))
            if verbose > 1:
                print_message('-- Log-likelihood = %.2e' % sum_log)
            elif verbose > 0:
                print('log-likelihood = %.2e' % sum_log)

        # We save the probability maps
        for i, pr in enumerate(ppr):
            prnii.get_data()[:] = pr
            prnnii.to_filename(os.path.join(path, 'tissue_pr%d.nii.gz' % i))

        brain = np.squeeze(np.argmax(ppr, axis=0) + 1).astype(mask_im.dtype) * mask_im

        # We'll find lesions by thresholding.
        if verbose > 0:
            print_message('- Lesion segmentation')
        flair_roi = flair[brain == 2]
        if verbose > 1:
            print_message('-- Threshold estimation')
        mu = flair_roi.mean()
        sigma = flair_roi.std()

        t = mu + alpha * sigma
        if verbose > 1:
            print_message('-- Threshold: %f (%f + %f * %f)' % (t, mu, alpha, sigma))

        wml = (sitk.GetArrayFromImage(flair) * mask_im) > t
        brain[wml] = brain.max() + 1

        masknii.get_data()[:] = brain
        masknii.to_filename(os.path.join(path, 'brain.nii.gz'))
