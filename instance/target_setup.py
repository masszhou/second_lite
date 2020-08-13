from utils.box_coder_torch import GroundBox3dCoderTorch
from utils.anchor_generator import AnchorGeneratorRange
from utils.region_similarity import get_similarity_rule
from utils.configure_tools import find_params

from typing import List, Dict
from model.type import BaseAnchorGenerator, BaseRegionSimilarityCalculator, BaseBoxCoder

REGISTERED_TARGET_SETUP = {}


def register_target_setup(cls, name=None):
    global REGISTERED_TARGET_SETUP
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_TARGET_SETUP, f"exist class: {REGISTERED_TARGET_SETUP}"
    REGISTERED_TARGET_SETUP[name] = cls
    return cls


def get_target_setup(name):
    global REGISTERED_TARGET_SETUP
    assert name in REGISTERED_TARGET_SETUP, f"available class: {REGISTERED_TARGET_SETUP}"
    return REGISTERED_TARGET_SETUP[name]


@register_target_setup
class TargetSetupKITTI:
    def __init__(self, class_cfg: Dict):
        self.cls_list: List[Dict] = [value for key, value in class_cfg.items()]

    def anchors_kitti(self) -> List[BaseAnchorGenerator]:
        """different object class should have different anchor setup
        """
        # e.g.
        # car = AnchorGeneratorRange(**find_params(AnchorGeneratorRange, car_cfg))
        # van = AnchorGeneratorRange(**find_params(AnchorGeneratorRange, van_cfg))
        # ...
        return [AnchorGeneratorRange(**find_params(AnchorGeneratorRange, cls_cfg)) for cls_cfg in self.cls_list]

    def nms_rules(self) -> List[Dict]:
        return [cls_cfg["nms"] for cls_cfg in self.cls_list]

    def classes_kitti(self) -> List[str]:
        return [cls_cfg["class_name"] for cls_cfg in self.cls_list]

    def similarity_rules_kitti(self) -> List[BaseRegionSimilarityCalculator]:
        return [get_similarity_rule(cls_cfg["similarity"]) for cls_cfg in self.cls_list]

    @staticmethod
    def box_coder() -> BaseBoxCoder:
        return GroundBox3dCoderTorch(smooth_dim=False, encode_angle_to_vector=False)


