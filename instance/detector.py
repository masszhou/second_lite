import torch
import numpy as np

from instance.builder_net import build_SECOND
from utils import visualization
from utils.configure_tools import parse_cfg, update_detect_range

from typing import List, Optional, Tuple
from model.type import SampleType, VoxelNetOutType


class SecondDetector:
    def __init__(self,
                 net_config_filepath: str,
                 class_config_filepath: str,
                 weight_filepath: str,
                 detect_range: Optional[Tuple[int, int, int, int]] = None):
        """
        :param detect_range: [xmin, ymin, xmax, ymax]
        """
        class_cfg = parse_cfg(class_config_filepath)
        net_cfg = parse_cfg(net_config_filepath)
        if detect_range is not None:
            class_cfg = update_detect_range(class_cfg, detect_range)
            net_cfg = update_detect_range(net_cfg, detect_range)

        # ======================================================
        # Build Network, Target Assigner and Voxel Generator
        # ======================================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_SECOND(net_cfg, class_cfg).to(self.device).eval()
        self.net.load_state_dict(torch.load(weight_filepath))

        # ======================================================
        # define voxel
        # ======================================================
        self.voxel_generator = self.net.voxel_generator
        middle_feature_extractor_downsample_factor = 8  # manual deduced from network layers
        grid_size = self.voxel_generator.grid_size
        feature_map_size = grid_size[:2] // middle_feature_extractor_downsample_factor
        feature_map_size = [*feature_map_size, 1][::-1]
        print("-> voxel information")
        print(f"grid_size = {self.voxel_generator.grid_size}")  # [1056 1280 40]  1056=(52.8-0)/0.05
        print(f"voxel_size = {self.voxel_generator.voxel_size}")  # [0.05 0.05 0.1 ]
        print(f"point_cloud_range = {self.voxel_generator.point_cloud_range}")  # [  0.  -32.   -3.   52.8  32.    1. ]
        print(f"feature map size = {feature_map_size}")

        # ======================================================
        # define how to encode objects
        # ======================================================
        self.target_assigner = self.net.target_assigner

        # ======================================================
        # Generate Anchors
        # ======================================================
        self.anchors = self.target_assigner.generate_anchors(feature_map_size)["anchors"]
        self.anchors = torch.tensor(self.anchors, dtype=torch.float32, device=self.device)
        self.anchors = self.anchors.view(1, -1, 7)

    def load_sample_from_points(self, points) -> SampleType:
        res = self.voxel_generator.generate(points[:, :4], max_voxels=90000)
        voxels, coords, num_points = res['voxels'], res['coordinates'], res['num_points_per_voxel']
        print("========== voxels.shape")
        print(voxels.shape)
        print("========== coords.shape")
        print(coords.shape)
        print("========== num_points")
        print(num_points.shape)

        # add a placeholder for batch index, from (x, y, z) to (0, x, y, z)
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)

        return {'anchors': self.anchors,
                'voxels': voxels,
                'num_points': num_points,
                'coordinates': coords,
                }

    def predict_on_points(self, pts_array: np.ndarray) -> List[VoxelNetOutType]:
        sample = self.load_sample_from_points(pts_array)
        pred = self.net(sample)
        return pred

    @staticmethod
    def visualize_bev(points: np.ndarray, boxes_lidar) -> np.ndarray:
        """
        @param: points: e.g. ndarray with shape (108348, 4)
        @param: boxes_lidar: center box format
        Returns: cv image
        """
        bev_map = SecondDetector.build_bev(points)
        bev_map = SecondDetector.draw_box_in_bev(bev_map, boxes_lidar)
        return bev_map

    @staticmethod
    def load_pts_from_file(pts_filepath: str, pts_dim: int = 4) -> np.ndarray:
        pts = np.fromfile(pts_filepath, dtype=np.float32, count=-1).reshape([-1, pts_dim])
        return pts

    @staticmethod
    def build_bev(points, vis_voxel_size=None, vis_point_range=None):
        if vis_voxel_size is None:
            vis_voxel_size = [0.1, 0.1, 0.1]
        if vis_point_range is None:
            vis_point_range = [-50, -30, -3, 50, 30, 1]  # [xmin, ymin, zmin, xmax, ymax, zmax]
        bev_map = visualization.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
        return bev_map

    @staticmethod
    def draw_box_in_bev(bev, boxes, vis_point_range=None, color=None):
        if vis_point_range is None:
            vis_point_range = [-50, -30, -3, 50, 30, 1]  # [xmin, ymin, zmin, xmax, ymax, zmax]
        if color is None:
            color = [0, 255, 0]
        bev_map = visualization.draw_box_in_bev(bev, vis_point_range, boxes, color, 2)
        return bev_map
