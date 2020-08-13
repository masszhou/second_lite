from model.type import BaseBoxCoder
from utils import box_ops_np


class GroundBaseBox3DCoder(BaseBoxCoder):
    def __init__(self, smooth_dim=False, encode_angle_to_vector=False, custom_ndim=0):
        super().__init__()
        self.smooth_dim = smooth_dim
        self.encode_angle_to_vector = encode_angle_to_vector
        self.custom_ndim = custom_ndim

    @property
    def code_size(self):
        res = 8 if self.encode_angle_to_vector else 7
        return self.custom_ndim + res

    def _encode(self, boxes, anchors):
        return box_ops_np.second_box_encode(boxes, anchors, self.encode_angle_to_vector, self.smooth_dim)

    def _decode(self, encodings, anchors):
        return box_ops_np.second_box_decode(encodings, anchors, self.encode_angle_to_vector, self.smooth_dim)