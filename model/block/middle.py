import numpy as np
from torch import nn

import spconv_lite as spconv
from utils.torchplus.tools import change_default_args
from utils.torchplus.nn import Empty


REGISTERED_MIDDLE_CLASSES = {}


def register_middle(cls, name=None):
    global REGISTERED_MIDDLE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MIDDLE_CLASSES, f"exist class: {REGISTERED_MIDDLE_CLASSES}"
    REGISTERED_MIDDLE_CLASSES[name] = cls
    return cls


def get_middle_class(name):
    global REGISTERED_MIDDLE_CLASSES
    assert name in REGISTERED_MIDDLE_CLASSES, f"available class: {REGISTERED_MIDDLE_CLASSES}"
    return REGISTERED_MIDDLE_CLASSES[name]


@register_middle
class SpMiddleFHDLite(nn.Module):
    def __init__(self,
                 spatial_shape,
                 use_norm=True,
                 num_input_features=3,
                 name='SpMiddleFHDLite'):
        """
        @param: output_shape: a dense shape, which feed to RPN
        @param: use_norm: if use batchnorm
        @param: num_input_features: depends on the output shape from VFE block
        @param: name:
        """
        super(SpMiddleFHDLite, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
        else:
            BatchNorm1d = Empty
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)

        # ToDO: add if condition for when need padding 3D tensor
        self.spatial_shape = np.array(spatial_shape) + [1, 0, 0]  # padding input shape e.g. [41 1280, 1056]
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SpConv3d(num_input_features, 16, 3, 2, padding=1),  # [1600, 1200, 41] -> [800, 600, 21]
            BatchNorm1d(16),
            nn.ReLU(),
            SpConv3d(16, 32, 3, 2, padding=1),  # [800, 600, 21] -> [400, 300, 11]
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, 2, padding=[0, 1, 1]),  # [400, 300, 11] -> [200, 150, 5]
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)),  # [200, 150, 5] -> [200, 150, 2]
            BatchNorm1d(64),
            nn.ReLU(),
        )
        # e.g
        # in: [#batch, c=128, D=1600, H=1200, W=41]
        # SpConv3d(ch_in=128, ch_out=16, k=3, stride=2, padding=1)
        # out: [#batch, c=16, D=800, H=600, W=21]
        # SpConv3d(16, 32, 3, 2, padding=1)
        # out: [#batch, c=32, D=400, H=300, W=11]
        # SpConv3d(32, 64, 3, 2, padding=[0, 1, 1])
        # out: [#batch, c=64, D=200, H=150, W=5]
        # SpConv3d(64, 64, k=(3, 1, 1), stride=(2, 1, 1))
        # out: [#batch, c=64, D=200, H=150, W=2]

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.spatial_shape, batch_size)  # e.g. [41 1280, 1056]
        ret = self.middle_conv(ret)  # e.g. [2, 160, 132]

        # ret.features = F.relu(ret.features)
        # print(self.middle_conv.fused())
        ret = ret.dense()  # shape=[1, 64, 2, 160, 132], -> [batch, channel, spatial_shape]

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)  # shape=[1, bev_c=128, bev_h=160, bev_w=132]
        return ret
