from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty


class BaseBoxCoder(object):
    """Abstract base class for box coder."""
    __metaclass__ = ABCMeta

    @abstractproperty
    def code_size(self):
        pass

    def encode(self, boxes, anchors):
        return self._encode(boxes, anchors)

    def decode(self, rel_codes, anchors):
        return self._decode(rel_codes, anchors)

    @abstractmethod
    def _encode(self, boxes, anchors):
        pass

    @abstractmethod
    def _decode(self, rel_codes, anchors):
        pass


class BaseAnchorGenerator:
    @property
    def class_name(self):
        raise NotImplementedError

    @property
    def num_anchors_per_localization(self):
        raise NotImplementedError

    def generate(self, feature_map_size):
        raise NotImplementedError

    @property
    def ndim(self):
        raise NotImplementedError


class BaseRegionSimilarityCalculator(object):
    """Abstract base class for 2d region similarity calculator."""
    __metaclass__ = ABCMeta

    def compare(self, boxes1, boxes2):
        """Computes matrix of pairwise similarity between BoxLists.

        This op (to be overriden) computes a measure of pairwise similarity between
        the boxes in the given BoxLists. Higher values indicate more similarity.

        Note that this method simply measures similarity and does not explicitly
        perform a matching.

        Args:
          boxes1: [N, 5] [x,y,w,l,r] tensor.
          boxes2: [M, 5] [x,y,w,l,r] tensor.

        Returns:
          a (float32) tensor of shape [N, M] with pairwise similarity score.
        """
        return self._compare(boxes1, boxes2)

    @abstractmethod
    def _compare(self, boxes1, boxes2):
        pass