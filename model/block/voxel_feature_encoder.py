import torch
from torch import nn
from typing import Optional, Tuple


REGISTERED_VFE_CLASSES = {}


def register_vfe(cls, name=None):
    global REGISTERED_VFE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_VFE_CLASSES, f"exist class: {REGISTERED_VFE_CLASSES}"
    REGISTERED_VFE_CLASSES[name] = cls
    return cls


def get_vfe_class(name):
    global REGISTERED_VFE_CLASSES
    assert name in REGISTERED_VFE_CLASSES, f"available class: {REGISTERED_VFE_CLASSES}"
    return REGISTERED_VFE_CLASSES[name]


@register_vfe
class SimpleVoxelRadius(nn.Module):
    """Simple voxel encoder. without neural network, only keep r, z and reflection feature.

    example:
    in: features -> [n_voxels=27632, max_point=30, x_y_z_ref=4]
    1. for each voxel, calculate mean x_y_z_ref
    2. calculate radius with (x,y)
    3. take (radius, z, ref) as a voxel feature, NO neural network
    sparse element output = [n_voxels=27632, r_z_ref=3]
    spatial output shape = channels, grid_dim
    """

    def __init__(self,
                 num_input_features: int = 4,
                 num_filters: int = 3,
                 use_norm: Optional[bool] = None,
                 with_distance: Optional[bool] = None,
                 voxel_size: Optional[Tuple[float, float, float]] = None,
                 pc_range: Optional[Tuple[float, float, float, float, float, float]] = None,
                 name: str = 'SimpleVoxelRadius'):

        super(SimpleVoxelRadius, self).__init__()
        self.num_input_features = num_input_features
        self.num_filters = num_filters  # equivalent to num_filters/out_channels from convolution op
        self.name = name

    def forward(self, features, num_voxels, coors):
        """
        @param: features: [concated_num_points, num_voxel_size, 3(4)]
        @param: num_voxels: [concated_num_points]
        @param: coors: [x,y,z] or [0, x, y, z], 0 is reserved for batch index
        Returns: e.g. [n_voxels=27632, r_z_ref=3]
        """
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :self.num_input_features].sum(dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        feature = torch.norm(points_mean[:, :2], p=2, dim=1, keepdim=True)
        # z is important for z position regression, but x, y is not.
        res = torch.cat([feature, points_mean[:, 2:self.num_input_features]], dim=1)
        return res
