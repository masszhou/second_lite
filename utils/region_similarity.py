# removed RotateIouSimilarity
from utils import box_ops_np
from model.type import BaseRegionSimilarityCalculator

__all__ = ['get_similarity_rule']

REGISTERED_SIMILARITY_RULES = {}


def register_similarity_rule(cls, name=None):
    global REGISTERED_SIMILARITY_RULES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_SIMILARITY_RULES, f"exist class: {REGISTERED_SIMILARITY_RULES}"
    REGISTERED_SIMILARITY_RULES[name] = cls
    return cls


def get_similarity_rule(name):
    global REGISTERED_SIMILARITY_RULES
    assert name in REGISTERED_SIMILARITY_RULES, f"available class: {REGISTERED_SIMILARITY_RULES}"
    return REGISTERED_SIMILARITY_RULES[name]


@register_similarity_rule
class NearestIouSimilarity(BaseRegionSimilarityCalculator):
    """Class to compute similarity based on the squared distance metric.

    This class computes pairwise similarity between two BoxLists based on the
    negative squared distance metric.
    """

    def _compare(self, boxes1, boxes2):
        """Compute matrix of (negated) sq distances.

        Args:
          boxlist1: BoxList holding N boxes.
          boxlist2: BoxList holding M boxes.

        Returns:
          A tensor with shape [N, M] representing negated pairwise squared distance.
        """
        boxes1_bv = box_ops_np.rbbox2d_to_near_bbox(boxes1)
        boxes2_bv = box_ops_np.rbbox2d_to_near_bbox(boxes2)
        ret = box_ops_np.iou_jit(boxes1_bv, boxes2_bv, eps=0.0)
        return ret


@register_similarity_rule
class DistanceSimilarity(BaseRegionSimilarityCalculator):
    """Class to compute similarity based on Intersection over Area (IOA) metric.

    This class computes pairwise similarity between two BoxLists based on their
    pairwise intersections divided by the areas of second BoxLists.
    """
    def __init__(self, distance_norm, with_rotation=False, rotation_alpha=0.5):
        self._distance_norm = distance_norm
        self._with_rotation = with_rotation
        self._rotation_alpha = rotation_alpha

    def _compare(self, boxes1, boxes2):
        """Compute matrix of (negated) sq distances.

        Args:
          boxlist1: BoxList holding N boxes.
          boxlist2: BoxList holding M boxes.

        Returns:
          A tensor with shape [N, M] representing negated pairwise squared distance.
        """
        return box_ops_np.distance_similarity(
            boxes1[..., [0, 1, -1]],
            boxes2[..., [0, 1, -1]],
            dist_norm=self._distance_norm,
            with_rotation=self._with_rotation,
            rot_alpha=self._rotation_alpha)