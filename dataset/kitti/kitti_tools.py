import pickle
from pathlib import Path
from typing import Union, Optional
import numpy as np
from tqdm import tqdm

from utils.box_ops_np import points_in_rbbox
from dataset.kitti import KittiDataset


def extract_groundtruth_objects(data_path: Union[Path, str],
                                info_path: Union[Path, str]):
    """ extract ground truth object' points cloud w.r.t 3d box label

    1. extract ground truth object' points cloud w.r.t 3d box label
    2. translate cropped object points cloud to origin
    3. save object points cloud into ".bin"
    4. create and append objects information into "kitti_dbinfos_train.pkl"
    """
    dataset = KittiDataset(info_path=info_path, root_path=data_path)

    root_path = Path(data_path)
    obj_pcl_save_path = root_path / 'gt_database'
    obj_pcl_save_path.mkdir(parents=True, exist_ok=True)
    db_info_save_path = root_path / "kitti_dbinfos_train.pkl"

    all_db_infos = {}

    for idx in tqdm(list(range(len(dataset))), total=len(dataset), desc="extract gt obj"):
        sensor_data = dataset.get_sensor_data(idx)
        image_idx = sensor_data["metadata"]["image_idx"]
        points = sensor_data["lidar"]["points"]
        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]
        names = annos["names"]
        group_dict = {}
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)  # reserved for group
        if "difficulty" in annos:
            difficulty = annos["difficulty"]
        else:
            difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)  # reserved for difficulty
        num_obj = gt_boxes.shape[0]
        point_indices = points_in_rbbox(points, gt_boxes)

        group_counter = 0
        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            filepath = obj_pcl_save_path / filename
            gt_points = points[point_indices[:, i]]  # e.g (152, 4), 152 point inside rotated bbox

            gt_points[:, :3] -= gt_boxes[i, :3]  # translate to origin, but not rotate to 0 degree ?
            with open(filepath, 'w') as f:
                gt_points.tofile(f)

            db_info = {
                "name": names[i],
                "path": str(filepath),
                "image_idx": image_idx,
                "gt_idx": i,
                "box3d_lidar": gt_boxes[i],
                "num_points_in_gt": gt_points.shape[0],
                "difficulty": difficulty[i],
            }

            local_group_id = group_ids[i]
            if local_group_id not in group_dict:
                group_dict[local_group_id] = group_counter
                group_counter += 1
            db_info["group_id"] = group_dict[local_group_id]

            if names[i] in all_db_infos:
                all_db_infos[names[i]].append(db_info)
            else:
                all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f"extract {len(v)} {k} objects infos")

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


def read_kitti_label(file_path):
    label_list = []
    try:
        with open(file_path, "r") as fs:
            for line in fs.readlines():
                line = line.strip().split(' ')
                for i in range(1, len(line)):
                    line[i] = float(line[i])

                box_3d_label = np.array(line[11:14]+[line[9], line[8], line[10], line[14]])  # x,y,z,w,h,l,yaw
                # x,y,z is ref camera coordinates in kitti label
                cls_name = line[0]
                label_list.append({"name": cls_name, "box": box_3d_label})
    except FileNotFoundError:
        print("label file not found")
    return label_list


class KittiCalibrationParser(object):
    """ Calibration and transformation

    1. 3d XYZ in <label>.txt are in rectified camera coord.
    2. 2d box xy are in image2 coord
    3. Points in <lidar>.bin are in Velodyne coord.

    y_image2 = P^2_rect * x_rect
    y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
    x_ref = Tr_velo_to_cam * x_velo
    x_rect = R0_rect * x_ref

    4. image2 coord:
        ----> x-axis (u)
        |
        |
        v y-axis (v)

    5. velodyne coord:
       front x, left y, up z

    6. rect/ref camera coord:
       right x, down y, front z

    7. camera 0 is the reference camera

    P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,  f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,  0,      1,      0            ]
             = K * [1|t]

    Note, P^2_rect is projection matrix after rectification! (co-plane)

    Ref
    1. KITTI paper: http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
    2. https://github.com/kuixu/kitti_object_vis/blob/master/kitti_util.py
    """

    def __init__(self, calib_filepath: Optional[str] = None):
        # reserve parameters for loading incomplete calibration information
        # camera intrinsic
        self.P = None
        self.c_u = None
        self.c_v = None
        self.f_u = None
        self.f_v = None
        self.b_x = None
        # Rigid transform from Velodyne coord to reference camera (camera0) coord
        self.V2C = None
        self.C2V = None
        # Rotation from reference camera coord to rect camera coord
        self.R0 = None

        self.calib_dict = {}

        if calib_filepath is not None:
            self.calib_dict = self.read_calib_file(calib_filepath)

    def __len__(self):
        return len(self.calib_dict)

    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        calib_dict = {}
        try:
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    line = line.rstrip()
                    if len(line) == 0: continue
                    key, value = line.split(':', 1)
                    # The only non-float values in these files are dates, which
                    # we don't care about anyway
                    try:
                        calib_dict[key] = np.array([float(x) for x in value.split()])
                    except ValueError:
                        pass
                self._parse(calib_dict)
        except FileNotFoundError:
            print(f"{filepath} not found")
        return calib_dict

    def _parse(self, calib_dict):
        # Projection matrix from rectified camera coord to image2 coord
        # P is projection matrix after rectification! (co-plane)
        # u        x
        # v = P R0 y
        # 1        z
        #          1
        #
        #     z          i -----> x(u)
        #    /           |
        #   c -- x   =>  |
        #   |            |
        #   y            y(v)
        #
        if "P2" in calib_dict.keys():
            self.P = calib_dict['P2']
            self.P = np.reshape(self.P, [3, 4])
            self.c_u = self.P[0, 2]
            self.c_v = self.P[1, 2]
            self.f_u = self.P[0, 0]
            self.f_v = self.P[1, 1]
            self.b_x = self.P[0, 3] / (-self.f_u)  # relative
            self.b_y = self.P[1, 3] / (-self.f_v)
        else:
            self.P = None
            self.P = None
            self.c_u = None
            self.c_v = None
            self.f_u = None
            self.f_v = None
            self.b_x = None
        # Rigid transform from Velodyne coord to reference camera (camera0) coord
        #
        #        z x        z
        #        |/        /
        #   y -- v    =>  c -- x
        #                 |
        #                 y
        #
        if "Tr_velo_to_cam" in calib_dict.keys():
            self.V2C = calib_dict['Tr_velo_to_cam']
            self.V2C = np.reshape(self.V2C, [3, 4])
            self.C2V = self.inverse_rigid_trans(self.V2C)
        else:
            self.V2C = None
            self.C2V = None

        # Rotation from reference camera coord to rect camera coord
        # (u,v)^T = P2 * R0 (x, y, z, 1)^T
        # (u, v) -> i-th camera image
        # (x, y, z, 1) -> reference camera coordinates, camera 0
        # R0 * (x, y, z, 1)^T -> i-th rectified camera coordinates, e.g. camera 2
        if "R0_rect" in calib_dict.keys():
            self.R0 = calib_dict['R0_rect']
            self.R0 = np.reshape(self.R0, [3, 3])
        else:
            self.R0 = None

    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        assert self.V2C is not None
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        assert self.C2V is not None
        rank = len(pts_3d_ref.shape)
        assert rank in [1, 2]
        if rank == 2:
            pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
            return np.dot(pts_3d_ref, np.transpose(self.C2V))
        else:
            pts_3d_ref = self.cart2hom(pts_3d_ref.reshape([-1, 3]))  # nx4
            return np.dot(pts_3d_ref, np.transpose(self.C2V)).flatten()

    def project_rect_to_ref(self, pts_3d_rect):
        """ Input and Output are nx3 points """
        assert self.R0 is not None
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        """ Input and Output are nx3 points """
        assert self.R0 is not None
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        """
        rank = len(pts_3d_rect.shape)
        assert rank in [1, 2]
        if rank == 2:
            pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
            return self.project_ref_to_velo(pts_3d_ref)
        else:
            pts_3d_ref = self.project_rect_to_ref(pts_3d_rect.reshape(-1, 3))
            pts_3d_velo = self.project_ref_to_velo(pts_3d_ref)
            return pts_3d_velo.flatten()

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        """ Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    def project_8p_to_4p(self, pts_2d):
        x0 = np.min(pts_2d[:, 0])
        x1 = np.max(pts_2d[:, 0])
        y0 = np.min(pts_2d[:, 1])
        y1 = np.max(pts_2d[:, 1])
        x0 = max(0, x0)
        # x1 = min(x1, proj.image_width)
        y0 = max(0, y0)
        # y1 = min(y1, proj.image_height)
        return np.array([x0, y0, x1, y1])

    def project_velo_to_4p(self, pts_3d_velo):
        """ Input: nx3 points in velodyne coord.
            Output: 4 points in image2 coord.
        """
        pts_2d_velo = self.project_velo_to_image(pts_3d_velo)
        return self.project_8p_to_4p(pts_2d_velo)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        """ Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        """
        assert None in [self.c_u, self.c_v, self.f_u, self.f_v, self.b_x, self.b_y]
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)

    def project_depth_to_velo(self, depth, constraint_box=True):
        depth_pt3d = self.get_depth_pt3d(depth)
        depth_UVDepth = np.zeros_like(depth_pt3d)
        depth_UVDepth[:, 0] = depth_pt3d[:, 1]
        depth_UVDepth[:, 1] = depth_pt3d[:, 0]
        depth_UVDepth[:, 2] = depth_pt3d[:, 2]
        # print("depth_pt3d:",depth_UVDepth.shape)
        depth_pc_velo = self.project_image_to_velo(depth_UVDepth)
        # print("dep_pc_velo:",depth_pc_velo.shape)
        if constraint_box:
            cbox = np.array([[0, 70.4], [-40, 40], [-3, 2]])  # todo: should be parameter
            depth_box_fov_inds = (depth_pc_velo[:, 0] < cbox[0][1]) & \
                                 (depth_pc_velo[:, 0] >= cbox[0][0]) & \
                                 (depth_pc_velo[:, 1] < cbox[1][1]) & \
                                 (depth_pc_velo[:, 1] >= cbox[1][0]) & \
                                 (depth_pc_velo[:, 2] < cbox[2][1]) & \
                                 (depth_pc_velo[:, 2] >= cbox[2][0])
            depth_pc_velo = depth_pc_velo[depth_box_fov_inds]
        return depth_pc_velo

    @staticmethod
    def inverse_rigid_trans(tr):
        """ Inverse a rigid body transform matrix (3x4 as [R|t])
           [R'|-R't; 0|1]
        """
        inv_tr = np.zeros_like(tr)  # 3x4
        inv_tr[0:3, 0:3] = np.transpose(tr[0:3, 0:3])
        inv_tr[0:3, 3] = np.dot(-np.transpose(tr[0:3, 0:3]), tr[0:3, 3])
        return inv_tr

    @staticmethod
    def get_depth_pt3d(depth):
        pt3d = []
        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):
                pt3d.append([i, j, depth[i, j]])
        return np.array(pt3d)