import itertools
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from generate_features import get_mask_voxels


def get_slices_bb(masks, patch_size, overlap, filtered=False, min_size=0):
    patch_half = map(lambda p_length: p_length // 2, patch_size)
    steps = map(lambda p_length: max(p_length - overlap, 1), patch_size)

    if type(masks) is list:
        min_bb = map(lambda mask: np.min(np.where(mask > 0), axis=-1), masks)
        min_bb = map(
            lambda min_bb_i: map(
                lambda (min_i, p_len): min_i + p_len,
                zip(min_bb_i, patch_half)
            ),
            min_bb
        )
        max_bb = map(lambda mask: np.max(np.where(mask > 0), axis=-1), masks)
        max_bb = map(
            lambda max_bb_i: map(
                lambda (max_i, p_len): max_i - p_len,
                zip(max_bb_i, patch_half)
            ),
            max_bb
        )

        dim_ranges = map(
            lambda (min_bb_i, max_bb_i): map(
                lambda t: np.concatenate([np.arange(*t), [t[1]]]),
                zip(min_bb_i, max_bb_i, steps)
            ),
            zip(min_bb, max_bb)
        )

        patch_slices = map(
            lambda dim_range: centers_to_slice(
                itertools.product(*dim_range), patch_half
            ),
            dim_ranges
        )

        if filtered:
            patch_slices = map(
                lambda (slices, mask): filter_size(slices, mask, min_size),
                zip(patch_slices, masks)
            )

    else:
        # Create bounding box and define
        min_bb = np.min(np.where(masks > 0), axis=-1)
        min_bb = map(
            lambda (min_i, p_len): min_i + p_len,
            zip(min_bb, patch_half)
        )
        max_bb = np.max(np.where(masks > 0), axis=-1)
        max_bb = map(
            lambda (max_i, p_len): max_i - p_len,
            zip(max_bb, patch_half)
        )

        dim_range = map(lambda t: np.arange(*t), zip(min_bb, max_bb, steps))
        patch_slices = centers_to_slice(
            itertools.product(*dim_range), patch_half
        )

        if filtered:
            patch_slices = filter_size(patch_slices, masks, min_size)

    return patch_slices


def centers_to_slice(voxels, patch_half):
    slices = map(
        lambda voxel: tuple(
            map(
                lambda (idx, p_len): slice(idx - p_len, idx + p_len),
                zip(voxel, patch_half)
            )
        ),
        voxels
    )
    return slices


def filter_size(slices, mask, min_size):
    filtered_slices = filter(
        lambda s_i: np.sum(mask[s_i] > 0) > min_size, slices
    )

    return filtered_slices


def get_balanced_slices(
        masks, patch_size, rois=None, min_size=0,
        neg_ratio=2
):
    # Init
    patch_half = map(lambda p_length: p_length // 2, patch_size)

    # Bounding box + not mask voxels
    if rois is None:
        min_bb = map(lambda mask: np.min(np.where(mask > 0), axis=-1), masks)
        max_bb = map(lambda mask: np.max(np.where(mask > 0), axis=-1), masks)
        bck_masks = map(np.logical_not, masks)
    else:
        min_bb = map(lambda mask: np.min(np.where(mask > 0), axis=-1), rois)
        max_bb = map(lambda mask: np.max(np.where(mask > 0), axis=-1), rois)
        bck_masks = map(
            lambda (m, roi): np.logical_and(m, roi.astype(bool)),
            zip(map(np.logical_not, masks), rois)
        )

    # The idea with this is to create a binary representation of illegal
    # positions for possible patches. That means positions that would create
    # patches with a size smaller than patch_size.
    # For notation, i = case; j = dimension
    max_shape = masks[0].shape
    mesh = get_mesh(max_shape)
    legal_masks = map(
        lambda (min_i, max_i): reduce(
            np.logical_and,
            map(
                lambda (m_j, min_ij, max_ij, p_ij, max_j): np.logical_and(
                    m_j >= max(min_ij, p_ij),
                    m_j <= min(max_ij, max_j - p_ij)
                ),
                zip(mesh, min_i, max_i, patch_half, max_shape)
            )
        ),
        zip(min_bb, max_bb)
    )

    # Filtering with the legal mask
    fmasks = map(
        lambda (m_i, l_i): np.logical_and(m_i, l_i), zip(masks, legal_masks)
    )
    fbck_masks = map(
        lambda (m_i, l_i): np.logical_and(m_i, l_i), zip(bck_masks, legal_masks)
    )

    lesion_voxels = map(get_mask_voxels, fmasks)
    bck_voxels = map(get_mask_voxels, fbck_masks)

    lesion_slices = map(
        lambda vox: centers_to_slice(vox, patch_half), lesion_voxels
    )
    bck_slices = map(
        lambda vox: centers_to_slice(vox, patch_half), bck_voxels
    )

    # Minimum size filtering for background
    fbck_slices = map(
        lambda (slices, mask): filter_size(slices, mask, min_size),
        zip(bck_slices, masks)
    )

    # Final slice selection
    patch_slices = map(
        lambda (pos_s, neg_s): pos_s + map(
            lambda idx: neg_s[idx],
            np.random.permutation(
                len(neg_s)
            )[:int(neg_ratio * len(pos_s))]
        ),
        zip(lesion_slices, fbck_slices)
    )

    return patch_slices


def get_mesh(shape):
    linvec = tuple(map(lambda s: np.linspace(0, s - 1, s), shape))
    mesh = np.stack(np.meshgrid(*linvec, indexing='ij')).astype(np.float32)
    return mesh


class GenericSegmentationCroppingDataset(Dataset):
    def __init__(
            self,
            cases, labels=None, masks=None, balanced=True,
            patch_size=32, neg_ratio=1, sampler=False
    ):
        # Init
        self.neg_ratio = neg_ratio
        # Image and mask should be numpy arrays
        self.sampler = sampler
        self.cases = cases
        self.labels = labels

        self.masks = masks

        data_shape = self.cases[0].shape

        if type(patch_size) is not tuple:
            patch_size = (patch_size,) * len(data_shape)
        self.patch_size = patch_size

        self.patch_slices = []
        if not self.sampler and balanced:
            if self.masks is not None:
                self.patch_slices = get_balanced_slices(
                    self.labels, self.patch_size, self.masks,
                    neg_ratio=self.neg_ratio
                )
            elif self.labels is not None:
                self.patch_slices = get_balanced_slices(
                    self.labels, self.patch_size, self.labels,
                    neg_ratio=self.neg_ratio
                )
            else:
                data_single = map(
                    lambda d: np.ones_like(
                        d[0] if len(d) > 1 else d
                    ),
                    self.cases
                )
                self.patch_slices = get_slices_bb(data_single, self.patch_size, 0)
        else:
            overlap = int(self.patch_size[0] // 1.1)
            if self.masks is not None:
                self.patch_slices = get_slices_bb(
                    self.masks, self.patch_size, overlap=overlap,
                    filtered=True
                )
            elif self.labels is not None:
                self.patch_slices = get_slices_bb(
                    self.labels, self.patch_size, overlap=overlap,
                    filtered=True
                )
            else:
                data_single = map(
                    lambda d: np.ones_like(
                        d[0] > np.min(d[0]) if len(d) > 1 else d
                    ),
                    self.cases
                )
                self.patch_slices = get_slices_bb(
                    data_single, self.patch_size, overlap=overlap,
                    filtered=True
                )
        self.max_slice = np.cumsum(map(len, self.patch_slices))

    def __getitem__(self, index):
        # We select the case
        case_idx = np.min(np.where(self.max_slice > index))
        case = self.cases[case_idx]

        slices = [0] + self.max_slice.tolist()
        patch_idx = index - slices[case_idx]
        case_slices = self.patch_slices[case_idx]

        # We get the slice indexes
        none_slice = (slice(None, None),)
        slice_i = case_slices[patch_idx]

        inputs = case[none_slice + slice_i].astype(np.float32)

        if self.labels is not None:
            labels = self.labels[case_idx].astype(np.uint8)
            target = np.expand_dims(labels[slice_i], 0)

            if self.sampler:
                return inputs, target, index
            else:
                return inputs, target
        else:
            return inputs, case_idx, slice_i

    def __len__(self):
        return self.max_slice[-1]


class WeightedSubsetRandomSampler(Sampler):

    def __init__(self, num_samples, sample_div=2, *args):
        super(WeightedSubsetRandomSampler, self).__init__(args)
        self.total_samples = num_samples
        self.num_samples = num_samples // sample_div
        self.weights = torch.tensor(
            [np.iinfo(np.int16).max] * num_samples, dtype=torch.double
        )
        self.indices = torch.randperm(num_samples)[:self.num_samples]

    def __iter__(self):
        return (i for i in self.indices.tolist())

    def __len__(self):
        return self.num_samples

    def update_weights(self, weights, idx):
        self.weights[idx] = weights.type_as(self.weights)

    def update(self):
        have = 0
        want = self.num_samples // 2
        n_rand = self.num_samples - want
        rand_indices = torch.randperm(self.total_samples)[:n_rand]
        p_ = self.weights.clone()
        p_[rand_indices] = 0
        indices = torch.empty(want, dtype=torch.long)
        while have < want:
            a = torch.multinomial(p_, want - have, replacement=True)
            b = a.unique()
            indices[have:have + b.size(-1)] = b
            p_[b] = 0
            have += b.size(-1)
        self.indices = torch.cat(
            (
                indices[torch.randperm(len(indices))],
                rand_indices
            )
        )
