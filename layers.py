from operator import mul
import torch
import itertools
from functools import reduce
import numpy as np
from torch import nn
from torch.nn import functional as F


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer pytorch

    The Layer can handle dense transforms that are meant to give a 'shift'
    from the current position. Therefore, a dense transform gives displacements
    (not absolute locations) at each voxel,

    This code is a reimplementation of
    https://github.com/marianocabezas/voxelmorph/tree/master/ext/neuron in
    pytorch with some liberties taken. The goal is to adapt the code to
    some kind of hybrid method to both do dense registration and mask tracking.
    """

    def __init__(
            self,
            interp_method='linear',
            linear_norm=False,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            **kwargs
    ):
        """
        Parameters:
            :param df_shape: Shape of the deformation field.
            :param interp_method: 'linear' or 'nearest'.
        """
        super().__init__()
        self.interp_method = interp_method
        self.device = device
        self.linear_norm = linear_norm

    def forward(self, inputs):
        """
        Transform (interpolation N-D volumes (features) given shifts at each
        location in pytorch. Essentially interpolates volume vol at locations
        determined by loc_shift.
        This is a spatial transform in the sense that at location [x] we now
        have the data from, [x + shift] so we've moved data.
        Parameters
            :param inputs: Input volume to be warped and deformation field.

            :return new interpolated volumes in the same size as df
        """

        # parse shapes
        n_inputs = len(inputs)
        if n_inputs > 2:
            vol, df, mesh = inputs
        else:
            vol, df = inputs
            mesh = None
        df_shape = df.shape[2:]
        final_shape = vol.shape[:2] + df_shape
        weights_shape = (vol.shape[0], 1) + df_shape
        nb_dims = len(df_shape)
        max_loc = map(lambda s: s - 1, vol.shape[2:])

        # location should be mesh and delta
        if n_inputs > 2:
            loc = [
                mesh[:, d, ...] + df[:, d, ...]
                for d in range(nb_dims)
            ]
        else:
            linvec = [torch.arange(0, s) for s in df_shape]
            mesh = [
                m_i.type(dtype=torch.float32) for m_i in torch.meshgrid(linvec)
            ]
            loc = [
                mesh[d].to(df.device) + df[:, d, ...]
                for d in range(nb_dims)
            ]
        loc = [
            torch.clamp(l, 0, m) for l, m in zip(loc, max_loc)
        ]

        # pre ind2sub setup
        d_size = np.cumprod((1,) + vol.shape[-1:2:-1])[::-1]

        # interpolate
        interp_vol = None
        if self.interp_method == 'linear':
            loc0 = map(torch.floor, loc)

            # clip values
            loc0lst = [
                torch.clamp(l, 0, m) for l, m in zip(loc0, max_loc)
            ]

            # get other end of point cube
            loc1 = [
                torch.clamp(l + 1, 0, m) for l, m in zip(loc0, max_loc)
            ]
            locs = [
                [f.type(torch.long) for f in loc0lst],
                [f.type(torch.long) for f in loc1]
            ]

            # compute the difference between the upper value and the original value
            # differences are basically 1 - (pt - floor(pt))
            #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
            diff_loc1 = [l1 - l for l1, l in zip(loc1, loc)]
            diff_loc1 = [torch.clamp(l, 0, 1) for l in diff_loc1]
            diff_loc0 = [1 - diff for diff in diff_loc1]
            weights_loc = [diff_loc1, diff_loc0]  # note reverse ordering since weights are inverse of diff.

            # go through all the cube corners, indexed by a ND binary vector
            # e.g. [0, 0] means this "first" corner in a 2-D "cube"
            cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
            norm_factor = nb_dims * len(cube_pts) / 2.0

            def get_point_value(point):
                subs = [locs[cd][i] for i, cd in enumerate(point)]
                loc_list_p = [s * l for s, l in zip(subs, d_size)]
                idx_p = torch.sum(torch.stack(loc_list_p, dim=0), dim=0)

                vol_val_flat = torch.stack(
                    [torch.stack(
                        [torch.take(vol_ij, idx_i) for vol_ij in vol_i],
                        dim=0
                    ) for vol_i, idx_i in zip(vol, idx_p)],
                    dim=0
                )

                vol_val = torch.reshape(vol_val_flat, final_shape)
                # get the weight of this cube_pt based on the distance
                # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
                # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
                wts_lst = [weights_loc[cd][i] for i, cd in enumerate(point)]
                if self.linear_norm:
                    wt = sum(wts_lst) / norm_factor
                else:
                    wt = reduce(mul, wts_lst)

                wt = torch.reshape(wt, weights_shape)
                return wt * vol_val

            values = map(get_point_value, cube_pts)
            interp_vol = torch.sum(torch.stack(values, dim=0), dim=0)

        elif self.interp_method == 'nearest':
            # clip values
            roundloc = [
                torch.clamp(l, 0, m).type(torch.long) for l, m in zip(
                    [torch.round(l) for l in loc], max_loc
                )
            ]

            # get values
            loc_list = [s * l for s, l in zip(roundloc, d_size)]
            idx = torch.sum(torch.stack(loc_list, dim=0), dim=0)
            interp_vol_flat = torch.stack(
                [torch.take(vol_i, idx_i) for idx_i, vol_i in zip(idx, vol)],
                dim=0
            )
            interp_vol = torch.reshape(interp_vol_flat, final_shape)

        return interp_vol


class SmoothingLayer(nn.Module):
    """
    N-D Smoothing layer pytorch

    The layer defines a trainable Gaussian smoothing kernel. While
    convolutional layers might learn such a kernel, the idea is to impose
    smoothing to the activations of the previous layer. The only parameter
    is the sigma value for the Gaussian kernel of a fixed size.
    """
    def __init__(
            self,
            length=5,
            init_sigma=0.5,
            trainable=False
    ):
        super().__init__()
        if trainable:
            self.sigma = nn.Parameter(
                torch.tensor(
                    init_sigma,
                    dtype=torch.float,
                    requires_grad=True
                )
            )
        else:
            self.sigma = torch.tensor(
                    init_sigma,
                    dtype=torch.float
                )
        self.length = length

    def forward(self, x):
        dims = len(x.shape) - 2
        assert dims <= 3, 'Too many dimensions for convolution'

        kernel_shape = (self.length,) * dims
        lims = map(lambda s: (s - 1.) / 2, kernel_shape)
        grid = map(
            lambda g: torch.tensor(g, dtype=torch.float, device=x.device),
            np.ogrid[tuple(map(lambda l: slice(-l, l + 1), lims))]
        )
        sigma_square = self.sigma * self.sigma
        k = torch.exp(
            -sum(map(lambda g: g*g, grid)) / (2. * sigma_square.to(x.device))
        )
        sumk = torch.sum(k)
        if sumk.tolist() > 0:
            k = k / sumk

        kernel = torch.reshape(k, (1,) * 2 + kernel_shape).to(x.device)
        final_kernel = kernel.repeat((x.shape[1],) * 2 + (1,) * dims)
        conv_f = [F.conv1d, F.conv2d, F.conv3d]
        padding = self.length / 2

        smoothed_x = conv_f[dims - 1](x, final_kernel, padding=padding)

        return smoothed_x
