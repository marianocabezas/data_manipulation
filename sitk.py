from __future__ import print_function
import SimpleITK as sitk
import os
import re
import numpy as np


def find_file(name, dirname):
    """

    :param name:
    :param dirname:
    :return:
    """
    result = filter(
        lambda x: not os.path.isdir(x) and re.search(name, x),
        os.listdir(dirname)
    )

    return result[0] if result else None


def print_current(reg_method, tf):
    """

    :param reg_method:
    :param tf:
    :return:
    """
    print('\t  MI (%d): %f\n\t  %s: [%s]' % (
        reg_method.GetOptimizerIteration(),
        reg_method.GetMetricValue(),
        tf.GetName(),
        ', '.join(['%s' % p for p in tf.GetParameters()])))


def itkresample(
        fixed,
        moving,
        transform,
        path=None,
        name=None,
        default_value=0.0,
        interpolation=sitk.sitkBSpline,
        verbose=0
):
    """

    :param fixed:
    :param moving:
    :param transform:
    :param path:
    :param name:
    :param default_value:
    :param interpolation:
    :param verbose:
    :return:
    """

    interpolation_dict = {
        'linear': sitk.sitkLinear,
        'bspline': sitk.sitkBSpline,
        'nn': sitk.sitkNearestNeighbor,
    }

    # Init
    if isinstance(fixed, basestring):
        fixed = sitk.ReadImage(fixed)
    elif isinstance(fixed, np.ndarray):
        fixed = sitk.GetImageFromArray(fixed)
    if isinstance(moving, basestring):
        moving = sitk.ReadImage(moving)
    elif isinstance(moving, np.ndarray):
        moving = sitk.GetImageFromArray(moving)

    if verbose > 1:
        print('\t  Image: ' + os.path.join(path, name + '.nii.gz'))

    if path is None or name is None or find_file(name + '.nii.gz', path) is None:
        interp_alg = interpolation if not isinstance(interpolation, basestring)\
            else interpolation_dict[interpolation]

        resampled = sitk.Resample(moving, fixed, transform, interp_alg, default_value)

        if path is not None and name is not None:
            sitk.WriteImage(resampled, os.path.join(path, name + '.nii.gz'))
    else:
        resampled = sitk.ReadImage(os.path.join(path, name + '.nii.gz'))

    return sitk.GetArrayFromImage(resampled)


def itkwarp(
        fixed,
        moving,
        field,
        path=None,
        name=None,
        default_value=0.0,
        interpolation=sitk.sitkBSpline,
        verbose=0,
):
    """

    :param fixed: Fixed image
    :param moving: Moving image
    :param field: Displacement field
    :param path:
    :param name:
    :param default_value:
    :param interpolation: interpolation function
    :param verbose:
    :return:
    """

    if isinstance(field, basestring):
        field = sitk.ReadImage(field)

    df_transform = sitk.DisplacementFieldTransform()
    df_transform.SetInverseDisplacementField(field)

    return itkresample(
        fixed, moving, df_transform,
        path, name, default_value, interpolation, verbose
    )


def itkn4(
        image,
        path=None,
        name=None,
        max_iters=400,
        levels=1,
        cast=sitk.sitkFloat32,
        verbose=1
):
    """

    :param image:
    :param path:
    :param name:
    :param max_iters:
    :param levels:
    :param cast:
    :param verbose:
    :return:
    """

    # Init
    if isinstance(image, basestring):
        image = sitk.ReadImage(image)
    elif isinstance(image, np.ndarray):
        image = sitk.GetImageFromArray(image)

    if verbose > 1:
        print('\t  Image: ' + os.path.join(path, name + '_corrected.nii.gz'))
    if path is None or name is None or find_file(name + '_corrected.nii.gz', path) is None:
        mask = sitk.OtsuThreshold(image, 0, 1, 200)
        image = sitk.Cast(image, cast)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([max_iters] * levels)
        output = corrector.Execute(image, mask)
        if name is not None and path is not None:
            sitk.WriteImage(output, os.path.join(path, name + '_corrected.nii.gz'))
        return sitk.GetArrayFromImage(output)


def itkhist_match(
        fixed,
        moving,
        path=None,
        name=None,
        histogram_levels=1024,
        match_points=7,
        mean_on=True,
        verbose=1
):
    """

    :param fixed:
    :param moving:
    :param path:
    :param name:
    :param histogram_levels:
    :param match_points:
    :param mean_on:
    :param verbose:
    :return:
    """

    # Init
    if isinstance(fixed, basestring):
        fixed = sitk.ReadImage(fixed)
    elif isinstance(fixed, np.ndarray):
        fixed = sitk.GetImageFromArray(fixed)
    if isinstance(moving, basestring):
        moving = sitk.ReadImage(moving)
    elif isinstance(moving, np.ndarray):
        moving = sitk.GetImageFromArray(moving)

    if verbose > 1:
        print('\t  Image: ' + os.path.join(path, name + '_corrected_matched.nii.gz'))
    if path is None or name is None or find_file(name + '_corrected_matched.nii.gz', path) is None:
        matched = sitk.HistogramMatching(moving, fixed, histogram_levels, match_points, mean_on)
        if name is not None and path is not None:
            sitk.WriteImage(matched, os.path.join(path, name + '_corrected_matched.nii.gz'))
        return sitk.GetArrayFromImage(matched)


def itksmoothing(image, path=None, name=None, sigma=0.5, sufix='_smoothed_subtraction.nii.gz', verbose=1):
    """

    :param image:
    :param path:
    :param name:
    :param sigma:
    :param sufix:
    :param verbose:
    :return:
    """

    # Init
    if isinstance(image, basestring):
        image = sitk.ReadImage(image)
    elif isinstance(image, np.ndarray):
        image = sitk.GetImageFromArray(image)

    # Gaussian smoothing
    gauss_filter = sitk.DiscreteGaussianImageFilter()
    gauss_filter.SetVariance(sigma * sigma)

    if verbose > 1:
        print('\t  Image: ' + os.path.join(path, name + sufix))
    if path is None or name is None or find_file(name + sufix, path) is None:
        smoothed = gauss_filter.Execute(image)
        sitk.WriteImage(smoothed, os.path.join(path, name + sufix))
    else:
        smoothed = sitk.ReadImage(os.path.join(path, name + sufix))
    return sitk.GetArrayFromImage(smoothed)


def itkrigid(
        fixed,
        moving,
        name='',
        number_bins=50,
        levels=3,
        steps=50,
        sampling=0.5,
        learning_rate=1.0,
        min_step=0.0001,
        max_step=0.2,
        relaxation_factor=0.5,
        cast= sitk.sitkFloat32,
        verbose=1
):
    """

    :param fixed:
    :param moving:
    :param name:
    :param number_bins:
    :param levels:
    :param steps:
    :param sampling:
    :param learning_rate:
    :param min_step:
    :param max_step:
    :param relaxation_factor:
    :param cast:
    :param verbose:
    :return:
    """
    # Init
    if isinstance(fixed, basestring):
        fixed = sitk.ReadImage(fixed)
    elif isinstance(fixed, np.ndarray):
        fixed = sitk.GetImageFromArray(fixed)
    if isinstance(moving, basestring):
        moving = sitk.ReadImage(moving)
    elif isinstance(moving, np.ndarray):
        moving = sitk.GetImageFromArray(moving)
    fixed_float32 = sitk.Cast(fixed, cast)
    moving_float32 = sitk.Cast(moving, cast)

    ''' Transformations '''
    initial_tf = sitk.CenteredTransformInitializer(
        fixed_float32,
        moving_float32,
        sitk.VersorRigid3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS
    )

    ''' Registration parameters '''
    registration = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=number_bins)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(sampling)
    registration.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=min_step,
        numberOfIterations=steps,
        maximumStepSizeInPhysicalUnits=max_step,
        relaxationFactor=relaxation_factor
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    smoothing_sigmas = range(levels - 1, -1, -1)
    shrink_factor = [2**i for i in smoothing_sigmas]
    registration.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factor)
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Connect all of the observers so that we can perform plotting during registration.
    if verbose > 0:
        print('\tRigid initial registration')
        registration.AddCommand(
            sitk.sitkMultiResolutionIterationEvent,
            lambda: print('\t > %s (%s) level %d' % (
                registration.GetName(),
                name,
                registration.GetCurrentLevel()
            ))
        )
    if verbose > 1:
        registration.AddCommand(
            sitk.sitkIterationEvent,
            lambda: print_current(registration, initial_tf)
        )

    # Initial versor optimisation
    registration.SetInitialTransform(initial_tf)
    registration.Execute(fixed_float32, moving_float32)

    return initial_tf


def itkaffine(
        fixed,
        moving,
        name='',
        initial_tf=None,
        number_bins=50,
        levels=3,
        steps=50,
        sampling=0.5,
        learning_rate=1.0,
        min_step=0.0001,
        max_step=0.2,
        relaxation_factor=0.5,
        cast=sitk.sitkFloat32,
        verbose=1
):
    """

    :param fixed:
    :param moving:
    :param name:
    :param initial_tf:
    :param number_bins:
    :param levels:
    :param steps:
    :param sampling:
    :param learning_rate:
    :param min_step:
    :param max_step:
    :param relaxation_factor:
    :param cast:
    :param verbose:
    :return:
    """

    # Init
    if isinstance(fixed, basestring):
        fixed = sitk.ReadImage(fixed)
    elif isinstance(fixed, np.ndarray):
        fixed = sitk.GetImageFromArray(fixed)
    if isinstance(moving, basestring):
        moving = sitk.ReadImage(moving)
    elif isinstance(moving, np.ndarray):
        moving = sitk.GetImageFromArray(moving)
    if initial_tf is None:
        initial_tf = itkrigid(fixed, moving, name, verbose=verbose)

    fixed_float32 = sitk.Cast(fixed, cast)
    moving_float32 = sitk.Cast(moving, cast)
    optimized_tf = sitk.AffineTransform(3)

    ''' Registration parameters '''
    registration = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=number_bins)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(sampling)
    registration.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=min_step,
        numberOfIterations=steps,
        maximumStepSizeInPhysicalUnits=max_step,
        relaxationFactor=relaxation_factor
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    smoothing_sigmas = range(levels - 1, -1, -1)
    shrink_factor = [2**i for i in smoothing_sigmas]
    registration.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factor)
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    '''Affine'''
    # Optimizer settings.
    registration.RemoveAllCommands()
    if verbose > 0:
        print('\tAffine registration')
        registration.AddCommand(
            sitk.sitkMultiResolutionIterationEvent,
            lambda: print('\t > %s (%s) level %d' % (
                registration.GetName(),
                name,
                registration.GetCurrentLevel()
            ))
        )
    if verbose > 1:
        registration.AddCommand(
            sitk.sitkIterationEvent,
            lambda: print_current(registration, optimized_tf)
        )

    registration.SetMovingInitialTransform(initial_tf)
    registration.SetInitialTransform(optimized_tf)

    registration.Execute(fixed_float32, moving_float32)

    final_tf = sitk.Transform(optimized_tf)
    final_tf.AddTransform(initial_tf)

    return final_tf


def itksubtraction(fixed, moving, path=None, name=None, verbose=1):
    """

    :param fixed:
    :param moving:
    :param path:
    :param name:
    :param verbose:
    :return:
    """

    # Init
    if isinstance(fixed, basestring):
        fixed = sitk.ReadImage(fixed)
    elif isinstance(fixed, np.ndarray):
        fixed = sitk.GetImageFromArray(fixed)
    if isinstance(moving, basestring):
        moving = sitk.ReadImage(moving)
    elif isinstance(moving, np.ndarray):
        moving = sitk.GetImageFromArray(moving)

    if verbose > 1:
        print('\t  Image: ' + os.path.join(path, name + '_subtraction.nii.gz'))

    if path is None or name is None or find_file(name + '_subtraction.nii.gz', path) is None:
        sub = sitk.Subtract(
            sitk.Cast(fixed, sitk.sitkFloat32),
            sitk.Cast(moving, sitk.sitkFloat32)
        )
        if path is not None and name is not None:
            sitk.WriteImage(sub, os.path.join(path, name + '_subtraction.nii.gz'))
    else:
        sub = sitk.ReadImage(os.path.join(path, name + '_subtraction.nii.gz'))

    return sitk.GetArrayFromImage(sub)


def itkdemons(
        fixed,
        moving,
        mask=None,
        path=None,
        name=None,
        steps=50,
        sigma=1.0,
        cast=sitk.sitkFloat32,
        verbose=1
):
    """

    :param fixed:
    :param moving:
    :param mask:
    :param path:
    :param name:
    :param steps:
    :param sigma:
    :param cast:
    :param verbose:
    :return:
    """
    # Init
    if isinstance(fixed, basestring):
        fixed = sitk.ReadImage(fixed)
    elif isinstance(fixed, np.ndarray):
        fixed = sitk.GetImageFromArray(fixed)
    if isinstance(moving, basestring):
        moving = sitk.ReadImage(moving)
    elif isinstance(moving, np.ndarray):
        moving = sitk.GetImageFromArray(moving)

    if mask is not None:
        fixed = sitk.Mask(fixed, mask)
        moving = sitk.Mask(moving, mask)
    fixed_float32 = sitk.Cast(fixed, cast)
    moving_float32 = sitk.Cast(moving, cast)

    deformation_name = name + '_multidemons_deformation.nii.gz'

    if verbose > 1:
        print('\t  Deformation: ' + os.path.join(path, deformation_name))

    if path is None or name is None or find_file(deformation_name, path) is None:
        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(steps)
        demons.SetStandardDeviations(sigma)

        if verbose > 1:
            demons.AddCommand(
                sitk.sitkIterationEvent,
                lambda: print('\t  Demons %d: %f' % (demons.GetElapsedIterations(), demons.GetMetric()))
            )

        deformation_field = demons.Execute(fixed_float32, moving_float32)
        sitk.WriteImage(deformation_field, os.path.join(path, deformation_name))
    else:
        deformation_field = sitk.ReadImage(os.path.join(path, deformation_name))

    return sitk.GetArrayFromImage(deformation_field)
