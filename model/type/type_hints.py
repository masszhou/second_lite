from typing import Dict, Union, List, Tuple, TypedDict, Optional
import numpy as np
import torch


class SampleType(TypedDict, total=False):
    """SampleType for prediction
    point cloud are saved in voxel format ( a sparse representation )
    """
    anchors: Union[torch.Tensor, np.ndarray]  # e.g. (feature_map=42240, xyzwlh_yaw=7)
    voxels: Union[torch.Tensor, np.ndarray]  # e.g. (non_empty_voxels=15000, max_points_per_voxel=5, xyzr=4)
    num_points: Union[torch.Tensor, np.ndarray]  # e.g. (15000, )
    coordinates: Union[torch.Tensor, np.ndarray]  # e.g. (15000, DHW=3)  D,H,W is voxel grid coordinates due to ZYX
    num_voxels: Union[torch.Tensor, np.ndarray]  # e.g. (1,) value=array([18391]), optional, for multi-gpu
    metadata: Dict  # e.g. {'image_idx': 11, 'image_shape': array([ 375, 1242], dtype=int32)}
    anchors_mask: Union[torch.Tensor, np.ndarray]


class TrainSampleType(SampleType):
    """extensions of SampleType for training
    add label information
    """
    # anchors:  array, e.g. shape = (non_empty_voxels=18391, max_points_per_voxel=5, xyzr=4)
    # voxels: array, e.g. shape = (18391, DHW=3), D,H,W is voxel grid coordinates due to ZYX
    # num_points:  array, e.g. shape = (18391,)
    # coordinates: array, e.g. shape = (18391, DHW=3), D,H,W is voxel grid coordinates due to ZYX
    # num_voxels: array, single value, optional, for multi-gpu
    metrics: Dict[str, float]  # e.g. {'voxel_gene_time': 0.0008378028869628906, 'prep_time': 0.026045799255371094}
    calib: Dict[str, np.ndarray]  # e.g. {"rect": ndarray 4x4, "Trv2c": ndarray 4x4, "P2": ndarray 4x4}
    gt_names: np.ndarray  # e.g. array(['Car', 'Car', 'Car', 'Car', 'Car'], dtype='<U10')
    labels: np.ndarray  # e.g. shape=(42240,)
    reg_targets: np.ndarray  # e.g. shape = (42240, 7)
    importance: np.ndarray  # e.g (42240,), all=1.0 for now


class RPNOutType(TypedDict):
    box_preds: torch.Tensor  # e.g. torch.Size([1, 2, 250, 250, 7])
    cls_preds: torch.Tensor  # e.g torch.Size([1, 2, 250, 250, 1])
    dir_cls_preds: torch.Tensor  # e.g. torch.Size([1, 2, 250, 250, 2])


class VoxelNetOutType(TypedDict):
    box3d_lidar: torch.Tensor  # rank=2 e.g. shape=[n,7] 7->(x, y, z, w, l, h, yaw)
    scores: torch.Tensor  # rank=1 e.g. shape=[n]
    label_preds: torch.Tensor  # rank=1 e.g. shape=[n]
    metadata: Optional[Dict]  # reserved


class LossesType(TypedDict, total=False):  # allow to miss some keys for now, todo: find a solution to avoid optional
    loss: Union[torch.Tensor, np.ndarray]
    cls_loss: Union[torch.Tensor, np.ndarray]
    loc_loss: Union[torch.Tensor, np.ndarray]
    cls_pos_loss: Union[torch.Tensor, np.ndarray]
    cls_neg_loss: Union[torch.Tensor, np.ndarray]
    cls_preds: Union[torch.Tensor, np.ndarray]
    cls_loss_reduced: Union[torch.Tensor, np.ndarray]
    loc_loss_reduced: Union[torch.Tensor, np.ndarray]
    cared: Union[torch.Tensor, np.ndarray]
    dir_loss_reduced: Union[torch.Tensor, np.ndarray]  # optional for use_direction_classifier


class ImageInfoType(TypedDict):
    """meta information for image data
    """
    image_idx: int
    image_path: str
    image_shape: Tuple[int, int]  # shape = (2,)


class PointCloudInfoType(TypedDict):
    """meta information for point cloud data
    """
    num_features: int  #  e.g. 4 for x,y,z,ref
    pcl_path: str  # e.g. 'training/velodyne/000000.bin'


class CalibrationInfoType(TypedDict, total=False):
    """calibration information
    todo: try to unify Kitti and lyft
    """
    # KITTI
    P0: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    P3: np.ndarray
    R0_rect: np.ndarray
    Tr_velo_to_cam: np.ndarray
    Tr_imu_to_velo: np.ndarray
    # lyft
    lidar2ego_translation: Tuple[float, float, float]  # x,y,z w.r.t the ego vehicle body frame.
    lidar2ego_rotation: Tuple[float, float, float, float]  # w, x, y, z w.r.t the ego vehicle body frame.
    ego2global_translation: Tuple[float, float, float]  # x, y, z w.r.t. log's map
    ego2global_rotation: Tuple[float, float, float, float]  # w, x, y, z w.r.t log's map


class AnnotationsInfoType(TypedDict, total=False):
    """annotation / label information
    for both 3D and 3D tasks
    allow to use partial properties
    """
    name: np.ndarray  # (n, ) with str, Describes the type of object: 'Car', 'Van',...
    truncated: np.ndarray  # (n, ), range in [0,1], truncated refers to the object leaving image boundaries
    occluded: np.ndarray  # (n, ) in {0,1,2,3}, 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
    alpha: np.ndarray  # (n, ) KITTI defined as Observation angle of object, ranging [-pi..pi]
    bbox: np.ndarray  # (n, 4) bounding boxes for image, for 2D task
    dimensions: np.ndarray  # (n, 3) in [m] todo: hwl for kitti, wlh for lyft, should be unified
    location: np.ndarray  # (n, 3), KITTI defined 3D object location x,y,z in camera coordinates in [m]
    rotation_y: np.ndarray  # (n, ) KITTI defined Rotation ry around Y-axis in CAMERA coordinates [-pi..pi] not LIDAR
    score: np.ndarray  # (n, ) Only for results: Float, indicating confidence in detection
    index: np.ndarray  # (n, )
    group_ids: np.ndarray  # (n, )
    difficulty: np.ndarray  # (n, )
    num_points_in_gt: np.ndarray  # (n, ) number of points in the 3D label box


class StandardInfoType(TypedDict):
    image: ImageInfoType


class KittiInfoType(TypedDict):
    """this type is supposed to be stored in mongodb in the future
    only contains meta data and information
    do not contain image or point clouds content
    """
    # todo: unify kitti with lyft format
    image: ImageInfoType
    point_cloud: PointCloudInfoType
    calib: CalibrationInfoType
    annos: AnnotationsInfoType
    metadata: Dict


class LyftInfoType(TypedDict):
    lidar_path: str
    cam_front_path: str
    sample_token: str
    sweeps: List
    lidar2ego_translation: Tuple[float, float, float]  # x,y,z w.r.t the ego vehicle body frame.
    lidar2ego_rotation: Tuple[float, float, float, float]  # w, x, y, z w.r.t the ego vehicle body frame.
    ego2global_translation: Tuple[float, float, float]  # x, y, z w.r.t. log's map
    ego2global_rotation: Tuple[float, float, float, float]  # w, x, y, z w.r.t log's map
    timestamp: float  # linux time stamp in [ms]
    gt_boxes: np.ndarray  # shape = (n_boxes, 7) x,y,z,w,l,h,yaw
    gt_names: np.ndarray  # shape = (n_boxes, ) with str content
    gt_velocity: np.ndarray  # shape = (n_boxes, 2), vx, vy
    num_lidar_pts: np.ndarray  # shape = (n_boxes, ) NOT used in Lyft, only for NuScense
    num_radar_pts: np.ndarray  # shape = (n_boxes, ) NOT used in Lyft, only for NuScense


class AnnotationsDataType(TypedDict):
    boxes: np.ndarray  # shape depends on task, for 3D task shape = (n, 7), for 2D task shape = (n, 4)
    names: List[str]


class LidarDataType(TypedDict, total=False):
    """
    with or without labels
    """
    type: str
    points: np.ndarray
    annotations: AnnotationsDataType


class ImageDataType(TypedDict):
    type: str
    data: np.ndarray
    datatype: str


class SensorDataType(TypedDict, total=False):
    lidar: LidarDataType
