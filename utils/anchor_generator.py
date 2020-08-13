import numpy as np
from utils import box_ops_np

from typing import Tuple, List
from model.type import BaseAnchorGenerator


class AnchorGeneratorRange(BaseAnchorGenerator):
    def __init__(self,
                 anchor_range: Tuple[float, float, float, float, float, float],
                 sizes: Tuple[float, float, float] = None,
                 rotations: Tuple[float, float] = None,
                 class_name: str = None,
                 match_threshold: float = -1.0,
                 unmatch_threshold: float = -1.0,
                 custom_values=(),
                 dtype=np.float32):
        """generate anchor template for each class
        different class has different size template, e.g. vehicle has larger box than pedestrian

        :param anchor_ranges: xmin, ymin, zmin, xmax, ymax, zmax in [m] x->front, y->left, z->up
        :param sizes: w,l,h in [m], note wlh order is different with feature lwh
        :param rotations: rotation anchors, default value is [0, pi/2]
        :param class_name: e.g. "Van" or "Car"
        :param match_threshold: e.g. 0.6 for "Car", 0.35 for "Pedestrian"
        :param unmatch_threshold: e.g. 0.45 for "Car", 0.2 for "Pedestrian"
        :param custom_values: reserved for e.g. velocities
        """
        super().__init__()
        # default values
        self._sizes = [1.6, 3.9, 1.56] if sizes is None else sizes
        self._rotations = [0, np.pi / 2] if rotations is None else rotations

        self._anchor_range = anchor_range
        self._dtype = dtype
        self._class_name = class_name
        self.match_threshold = match_threshold
        self.unmatch_threshold = unmatch_threshold
        self._custom_values = custom_values

    @property
    def class_name(self):
        return self._class_name

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    def generate(self, feature_map_size: List[int]) -> np.ndarray:
        # e.g. feature_map_size=[1,160,132]
        # res is ndarray with shape [1, 160, 132, 1, 2, 7]
        res = box_ops_np.create_anchors_3d_range(
            feature_map_size, self._anchor_range, self._sizes,
            self._rotations, self._dtype)

        if len(self._custom_values) > 0:
            custom_ndim = len(self._custom_values)
            custom = np.zeros([*res.shape[:-1], custom_ndim])
            custom[:] = self._custom_values
            res = np.concatenate([res, custom], axis=-1)
        return res

    @property
    def ndim(self):
        return 7 + len(self._custom_values)

    @property
    def custom_ndim(self):
        return len(self._custom_values)