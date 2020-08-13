import model.block as block
import model.net as net
from spconv_lite.utils import VoxelGeneratorV2
from utils.target_assigner import TargetAssigner
from instance.target_setup import get_target_setup
from utils.configure_tools import find_params

from typing import Tuple, Dict


def build_voxel_generator(voxel_cfg):
    voxel_generator = VoxelGeneratorV2(**find_params(VoxelGeneratorV2, voxel_cfg))
    return voxel_generator


def build_target_assigner(target_setup):
    target_assigner = TargetAssigner(
        box_coder=target_setup.box_coder(),  # e.g. like SSD regression
        anchor_generators=target_setup.anchors_kitti(),
        feature_map_sizes=[],  # can be deduced from voxel size, used for define anchor
        positive_fraction=-1,
        sample_size=512,
        region_similarity_calculators=target_setup.similarity_rules_kitti(),
        classes=target_setup.classes_kitti(),
        assign_per_class=True)
    return target_assigner


def build_vfe_block(vfe_cfg):
    """
    @param: module_name: e.g. SimpleVoxel has no NN inside, only calculcate means per voxel
    @param: num_filters: NO USE for SimpleVoxel, but is a hint for middle block
    @param: num_input_features: 4-> x, y, z, ref
    @return: Voxel feature extractor block
    """
    module_name = vfe_cfg["module_name"]
    return block.get_vfe_class(module_name)(num_input_features=vfe_cfg["num_input_features"],
                                            num_filters=vfe_cfg["num_filters"])


def build_middle_block(middle_cfg,
                       voxel_gen,
                       vfe):
    """
    @param: module_name:
    @param: num_input_features: "SimpleVoxelRadius" has output [n_voxel, rz_ref=3]
    @param: vfe:
    @param: voxel_gen:
    """
    module_name = middle_cfg["module_name"]
    grid_size = voxel_gen.grid_size  # e.g. [1056 1280 40]
    # estimate the dense shape of VFE output, prepare for sparseConv3D()
    input_spatial_shape = grid_size[::-1].tolist()  # e.g [1, 40, 1280, 1056, 3]

    return block.get_middle_class(module_name)(input_spatial_shape,
                                               use_norm=True,
                                               num_input_features=vfe.num_filters)


def build_rpn_block(rpn_cfg, target_assigner):
    module_name = rpn_cfg["module_name"]
    return block.get_rpn_class(module_name)(num_anchor_per_loc=target_assigner.num_anchors_per_location,
                                            box_code_size=target_assigner.box_coder.code_size,
                                            **find_params(block.RPNNoHeadBase.__init__, rpn_cfg))


def build_SECOND(net_cfg: Dict, class_cfg: Dict):

    voxel_cfg = net_cfg["VoxelGenerator"]
    voxel_generator = build_voxel_generator(voxel_cfg)

    target_setup = get_target_setup("TargetSetupKITTI")(class_cfg)
    target_assigner = build_target_assigner(target_setup)

    vfe_cfg = net_cfg["VoxelNet"]["vfe_module"]
    vfe_block = build_vfe_block(vfe_cfg)

    middle_cfg = net_cfg["VoxelNet"]["middle_module"]
    middle_block = build_middle_block(middle_cfg, voxel_generator, vfe_block)

    rpn_cfg = net_cfg["VoxelNet"]["rpn_module"]
    rpn_block = build_rpn_block(rpn_cfg, target_assigner)

    nms_rules = target_setup.nms_rules()
    nms_pre_max_sizes = [c["nms_pre_max_size"] for c in nms_rules]  # [1000]
    nms_post_max_sizes = [c["nms_post_max_size"] for c in nms_rules]
    nms_score_thresholds = [c["nms_score_threshold"] for c in nms_rules]
    nms_iou_thresholds = [c["nms_iou_threshold"] for c in nms_rules]

    second_net = net.VoxelNet(num_class=4,
                              num_input_features=4,
                              voxel_generator=voxel_generator,
                              target_assigner=target_assigner,
                              vfe_block=vfe_block,
                              middle_block=middle_block,
                              rpn_block=rpn_block,
                              use_rotate_nms=False,
                              multiclass_nms=False,
                              nms_pre_max_sizes=nms_pre_max_sizes,
                              nms_post_max_sizes=nms_post_max_sizes,
                              nms_score_thresholds=nms_score_thresholds,
                              nms_iou_thresholds=nms_iou_thresholds)
    return second_net
