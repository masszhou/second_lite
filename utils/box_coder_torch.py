from utils.box_coder_np import GroundBaseBox3DCoder
from utils import box_ops_torch


class GroundBox3dCoderTorch(GroundBaseBox3DCoder):
    def encode_torch(self, boxes, anchors):
        return box_ops_torch.second_box_encode(boxes, anchors, self.encode_angle_to_vector, self.smooth_dim)

    def decode_torch(self, boxes, anchors):
        return box_ops_torch.second_box_decode(boxes, anchors, self.encode_angle_to_vector, self.smooth_dim)
