import pickle
import numpy as np
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage import io
from tqdm import tqdm
from typing import List, Dict

from dataset.base import AbstractDataset
from dataset.base import register_dataset
from utils import box_ops_np
# from utils.eval import get_official_eval_result
# from utils.eval import get_coco_eval_result
from model.type import KittiInfoType
from model.type import TrainSampleType


@register_dataset
class KittiDataset(AbstractDataset):
    NumPointFeatures = 4  # input dim = x,y,z,ref

    def __init__(self,
                 root_path,  # "/media/zzhou/data-KITTI/object"
                 info_path,  # "/media/zzhou/data-KITTI/object/kitti_infos_train.pkl"
                 class_names=None,
                 prep_func=None):
        assert info_path is not None
        self._root_path = Path(root_path)
        with open(info_path, 'rb') as f:
            self._kitti_infos = pickle.load(f)  # type: List[KittiInfoType]
            # {'image': {'image_idx': 0,
            #            'image_path': 'training/image_2/000000.png',
            #            'image_shape': array([370, 1224], dtype=int32)},
            #  'point_cloud': {'num_features': 4,
            #                  'velodyne_path': 'training/velodyne/000000.bin'},
            #  'calib': {'P0': array(),  # 4x4
            #            'P1': array(),  # 4x4
            #            'P2': array(),  # 4x4
            #            'P3': array(),  # 4x4
            #            'R0_rect': array(),  # 4x4
            #            'Tr_velo_to_cam': array(),  # 4x4
            #            'Tr_imu_to_velo': array()  # 4x4
            #            },
            #  'annos': {'name': array(['Pedestrian'], dtype='<U10'),
            #            'truncated': array([0.]),
            #            'occluded': array([0]),
            #            'alpha': array([-0.2]),
            #            'bbox': array([[712.4, 143., 810.73, 307.92]]),
            #            'dimensions': array([[1.2, 1.89, 0.48]]),
            #            'location': array([[1.84, 1.47, 8.41]]),
            #            'rotation_y': array([0.01]),
            #            'score': array([0.]),
            #            'index': array([0], dtype=int32),
            #            'group_ids': array([0], dtype=int32),
            #            'difficulty': array([0], dtype=int32),
            #            'num_points_in_gt': array([256], dtype=int32)
            #            }
            # }
        print("remain number of infos:", len(self._kitti_infos))
        self._class_names = class_names
        self._prep_func = prep_func

    def __len__(self):
        return len(self._kitti_infos)  # e.g. 3712

    def __getitem__(self, idx) -> TrainSampleType:
        """load sensor data and transform to voxel data (a sparse representation)
        :return
        {
        "voxels": array, e.g. shape = (non_empty_voxels=18391, max_points_per_voxel=5, xyzr=4)
        "num_points": array, e.g. shape = (18391,)
        "coordinates": array, e.g. shape = (18391, DHW=3), D,H,W is voxel grid coordinates due to ZYX
        "num_voxels": array, single value, optional, for multi-gpu
        "anchors": array, e.g. shape = (42240, 7)
        "metrics": Dict, e.g. {'voxel_gene_time': 0.001383066177368164, 'prep_time': 6.540638446807861}
        "calib": Dict[str, darray]  e.g. {"rect": ndarray 4x4, "Trv2c": ndarray 4x4, "P2": ndarray 4x4}
        "gt_names": array, e.g. array(['Car', 'Car', 'Car', 'Car', 'Car'], dtype='<U10')
        "labels": array, e.g. shape=(42240,)
        "reg_targets" array, e.g. shape = (42240, 7)
        "importance": array,  e.g (42240,), all=1.0 for now
        "metadata": Dict,  e.g. {'image_idx': 11, 'image_shape': array([ 375, 1242], dtype=int32)}
        }
        """
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict=input_dict)
        example["metadata"] = {}
        if "image_idx" in input_dict["metadata"]:
            example["metadata"] = input_dict["metadata"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example

    def get_sensor_data(self, query: int):
        """load sensor data and assign label values w.r.t. meta information
        :return:
        {
        'lidar': {'type': 'lidar',
                   'points': array([[ 5.2440e+01,  1.0500e-01,  1.9820e+00,  0.0000e+00],
                                   [ 5.1906e+01,  2.6700e-01,  1.9640e+00,  0.0000e+00],
                                   [ 5.1965e+01,  4.3100e-01,  1.9660e+00,  7.0000e-02],
                                   ...,
                                   [ 6.3280e+00, -4.1000e-02, -1.6520e+00,  2.0000e-01],
                                   [ 6.3190e+00, -2.1000e-02, -1.6500e+00,  2.3000e-01],
                                   [ 6.3260e+00, -1.0000e-03, -1.6520e+00,  1.9000e-01]], dtype=float32),
                   'annotations': {'boxes': array([[12.70835552,-5.04503378, -0.47577397,  0.41999999,  1.03999996, 1.89999998,  0.68000001],
                                                  [13.71911654, -5.40414864, -0.55401332,  0.5       ,  0.89999998, 1.87      ,  0.67000002],
                                                  [26.92940758,  4.97040204, -0.64137194,  1.57000005,  3.82999992, 1.86000001,  1.54999995],
                                                  [34.36796531, -2.18102056, -0.41917181,  0.52999997,  0.94999999, 1.77999997, -0.33000001],
                                                  [ 4.42080725,  5.13780446, -1.07483279,  1.46000004,  3.70000005, 1.5       ,  1.55999994],
                                                  [16.24044422,  7.9400693 , -0.99171075,  0.56999999,  0.41      ,1.53999996,  1.57000005]]),
                                  'names': array(['Pedestrian', 'Pedestrian', 'Car', 'Pedestrian', 'Car', 'Pedestrian'], dtype='<U10')
                                  }
                 },
        'metadata': {'image_idx': 11,
                     'image_shape': array([ 375, 1242], dtype=int32)
                     },
        'calib': {'rect': array([[ 0.9999239 ,  0.00983776, -0.00744505,  0.        ],
                                 [-0.0098698 ,  0.9999421 , -0.00427846,  0.        ],
                                 [ 0.00740253,  0.00435161,  0.9999631 ,  0.        ],
                                 [ 0.        ,  0.        ,  0.        ,  1.        ]]),
                  'Trv2c': array([[ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                                  [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
                                  [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
                                  [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]),
                  'P2': array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
                               [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                               [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03],
                               [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]])
                 },
        'cam': {'annotations': {'boxes': array([[883.68, 144.15, 937.35, 259.01],
                                                [873.7 , 152.1 , 933.44, 256.07],
                                                [444.29, 171.04, 504.95, 225.82],
                                                [649.28, 168.1 , 664.61, 206.4 ],
                                                [  0.  , 217.12,  85.92, 374.  ],
                                                [240.35, 190.31, 268.02, 261.61]]),
                                'names': array(['Pedestrian', 'Pedestrian', 'Car', 'Pedestrian', 'Car', 'Pedestrian'], dtype='<U10')
                                }
               }
        }
        """
        read_image = False
        idx = query
        if isinstance(query, dict):
            read_image = "cam" in query
            assert "lidar" in query
            idx = query["lidar"]["idx"]
        info = self._kitti_infos[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "image_idx": info["image"]["image_idx"],
                "image_shape": info["image"]["image_shape"],
            },
            "calib": None,
            "cam": {}
        }

        pc_info = info["point_cloud"]
        velo_path = Path(pc_info['velodyne_path'])

        if not velo_path.is_absolute():
            velo_path = Path(self._root_path) / pc_info['velodyne_path']
        velo_reduced_path = velo_path.parent.parent / (velo_path.parent.stem + '_reduced') / velo_path.name

        if velo_reduced_path.exists():
            velo_path = velo_reduced_path

        points = np.fromfile(str(velo_path), dtype=np.float32, count=-1).reshape([-1, self.NumPointFeatures])
        res["lidar"]["points"] = points
        image_info = info["image"]
        image_path = image_info['image_path']

        if read_image:
            image_path = self._root_path / image_path
            with open(str(image_path), 'rb') as f:
                image_str = f.read()
            res["cam"] = {
                "type": "camera",
                "data": image_str,
                "datatype": image_path.suffix[1:],
            }
        calib = info["calib"]
        calib_dict = {
            'rect': calib['R0_rect'],
            'Trv2c': calib['Tr_velo_to_cam'],
            'P2': calib['P2'],
        }
        res["calib"] = calib_dict
        if 'annos' in info:
            annos = info['annos']
            # we need other objects to avoid collision when sample
            annos = remove_dontcare(annos)
            locs = annos["location"]
            dims = annos["dimensions"]
            rots = annos["rotation_y"]
            gt_names = annos["name"]
            # rots = np.concatenate([np.zeros([locs.shape[0], 2], dtype=np.float32), rots], axis=1)
            gt_boxes = np.concatenate([locs, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            calib = info["calib"]
            gt_boxes = box_ops_np.box_camera_to_lidar(gt_boxes, calib["R0_rect"], calib["Tr_velo_to_cam"])

            # only center format is allowed. so we need to convert
            # kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]
            box_ops_np.change_box3d_center_(gt_boxes, [0.5, 0.5, 0], [0.5, 0.5, 0.5])
            res["lidar"]["annotations"] = {
                'boxes': gt_boxes,
                'names': gt_names,
            }
            res["cam"]["annotations"] = {
                'boxes': annos["bbox"],
                'names': gt_names,
            }

        return res

    def evaluation(self, detections, output_dir):
        """
        detection
        When you want to eval your own dataset, you MUST set correct
        the z axis and box z center.
        If you want to eval by my KITTI eval function, you must
        provide the correct format annotations.
        ground_truth_annotations format:
        {
            bbox: [N, 4], if you fill fake data, MUST HAVE >25 HEIGHT!!!!!!
            alpha: [N], you can use -10 to ignore it.
            occluded: [N], you can use zero.
            truncated: [N], you can use zero.
            name: [N]
            location: [N, 3] center of 3d box.
            dimensions: [N, 3] dim of 3d box.
            rotation_y: [N] angle.
        }
        all fields must be filled, but some fields can fill
        zero.
        """
        # if "annos" not in self._kitti_infos[0]:
        #     return None
        # gt_annos = [info["annos"] for info in self._kitti_infos]
        # dt_annos = self.convert_detection_to_kitti_annos(detections)
        # # firstly convert standard detection to kitti-format dt annos
        # z_axis = 1  # KITTI camera format use y as regular "z" axis.
        # z_center = 1.0  # KITTI camera box's center is [0.5, 1, 0.5]
        # # for regular raw lidar data, z_axis = 2, z_center = 0.5.
        # result_official_dict = get_official_eval_result(
        #     gt_annos,
        #     dt_annos,
        #     self._class_names,
        #     z_axis=z_axis,
        #     z_center=z_center)
        # result_coco = get_coco_eval_result(
        #     gt_annos,
        #     dt_annos,
        #     self._class_names,
        #     z_axis=z_axis,
        #     z_center=z_center)
        # return {"results": {"official": result_official_dict["result"],
        #                     "coco": result_coco["result"],
        #                     },
        #         "detail": {"eval.kitti": {"official": result_official_dict["detail"],
        #                                   "coco": result_coco["detail"]}
        #                    },
        #         }
        return None

    def convert_detection_to_kitti_annos(self, detection):
        class_names = self._class_names
        det_image_idxes = [det["metadata"]["image_idx"] for det in detection]
        gt_image_idxes = [
            info["image"]["image_idx"] for info in self._kitti_infos
        ]
        annos = []
        for i in range(len(detection)):
            det_idx = det_image_idxes[i]
            det = detection[i]
            # info = self._kitti_infos[gt_image_idxes.index(det_idx)]
            info = self._kitti_infos[i]
            calib = info["calib"]
            rect = calib["R0_rect"]
            Trv2c = calib["Tr_velo_to_cam"]
            P2 = calib["P2"]
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            if final_box_preds.shape[0] != 0:
                final_box_preds[:, 2] -= final_box_preds[:, 5] / 2
                box3d_camera = box_np_ops.box_lidar_to_camera(final_box_preds, rect, Trv2c)
                locs = box3d_camera[:, :3]
                dims = box3d_camera[:, 3:6]
                angles = box3d_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = box_np_ops.center_to_corner_box3d(locs, dims, angles, camera_box_origin, axis=1)
                box_corners_in_image = box_np_ops.project_to_image(box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = np.min(box_corners_in_image, axis=1)
                maxxy = np.max(box_corners_in_image, axis=1)
                bbox = np.concatenate([minxy, maxxy], axis=1)
            anno = get_start_result_anno()
            num_example = 0
            box3d_lidar = final_box_preds
            for j in range(box3d_lidar.shape[0]):
                image_shape = info["image"]["image_shape"]
                if bbox[j, 0] > image_shape[1] or bbox[j, 1] > image_shape[0]:
                    continue
                if bbox[j, 2] < 0 or bbox[j, 3] < 0:
                    continue
                bbox[j, 2:] = np.minimum(bbox[j, 2:], image_shape[::-1])
                bbox[j, :2] = np.maximum(bbox[j, :2], [0, 0])
                anno["bbox"].append(bbox[j])
                # convert center format to kitti format
                # box3d_lidar[j, 2] -= box3d_lidar[j, 5] / 2
                anno["alpha"].append(-np.arctan2(-box3d_lidar[j, 1], box3d_lidar[j, 0]) + box3d_camera[j, 6])
                anno["dimensions"].append(box3d_camera[j, 3:6])
                anno["location"].append(box3d_camera[j, :3])
                anno["rotation_y"].append(box3d_camera[j, 6])

                anno["name"].append(class_names[int(label_preds[j])])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(scores[j])

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(empty_result_anno())
            num_example = annos[-1]["name"].shape[0]
            annos[-1]["metadata"] = det["metadata"]
        return annos


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def _calculate_num_points_in_gt(data_path: str,
                                infos: List[KittiInfoType],
                                relative_path: bool,
                                remove_outside=True,
                                num_features=4):
    """ calculate how many points are inside a 3D label box
    """
    for info in infos:
        pc_info = info["point_cloud"]
        image_info = info["image"]
        calib = info["calib"]
        if relative_path:
            v_path = str(Path(data_path) / pc_info["velodyne_path"])
        else:
            v_path = pc_info["velodyne_path"]
        points_v = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']
        if remove_outside:
            # remove points outside camera frustum, because KITTI dataset only labeled object inside camera frustum !
            points_v = box_ops_np.remove_outside_points(points_v, rect, Trv2c, P2, image_info["image_shape"])

        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        gt_boxes_lidar = box_ops_np.box_camera_to_lidar(gt_boxes_camera, rect, Trv2c)
        indices = box_ops_np.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate([num_points_in_gt, -np.ones([num_ignored])])
        annos["num_points_in_gt"] = num_points_in_gt.astype(np.int32)


def create_kitti_info_file(data_path, save_path=None, relative_path=True):
    """ summary data as a psudo database
    """
    imageset_folder = Path(__file__).resolve().parent / "KittiSets"
    train_img_ids = _read_imageset_file(str(imageset_folder / "train.txt"))
    val_img_ids = _read_imageset_file(str(imageset_folder / "val.txt"))
    test_img_ids = _read_imageset_file(str(imageset_folder / "test.txt"))

    print("Generate info. this may take several minutes.")
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    kitti_infos_train = get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=train_img_ids,
        relative_path=relative_path,
        op_name="build_train_info")  # type: List[Dict]
    _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
    filename = save_path / 'kitti_infos_train.pkl'
    print(f"Kitti info train file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)

    kitti_infos_val = get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=val_img_ids,
        relative_path=relative_path,
        op_name="build_val_info")
    _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    filename = save_path / 'kitti_infos_val.pkl'
    print(f"Kitti info val file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    filename = save_path / 'kitti_infos_trainval.pkl'
    print(f"Kitti info trainval file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)

    kitti_infos_test = get_kitti_image_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        image_ids=test_img_ids,
        relative_path=relative_path,
        op_name="build_test_info")
    filename = save_path / 'kitti_infos_test.pkl'
    print(f"Kitti info test file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)


def remove_dontcare(image_anno):
    img_filtered_annotations = {}
    relevant_annotation_indices = [i for i, x in enumerate(image_anno['name']) if x != "DontCare"]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (image_anno[key][relevant_annotation_indices])
    return img_filtered_annotations


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def get_kitti_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True,
                         op_name=None) -> List[KittiInfoType]:
    """
    KITTI annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 4}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(idx, path, training, relative_path)
        image_info['image_path'] = get_image_path(idx, path, training, relative_path)
        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info["image"] = image_info
        info["point_cloud"] = pc_info
        if calib:
            calib_path = get_calib_path(idx, path, training, relative_path=False)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]]).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
            R0_rect = np.array([float(info) for info in lines[4].split(' ')[1:10]]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = np.array([float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4])
            Tr_imu_to_velo = np.array([float(info) for info in lines[6].split(' ')[1:13]]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo
            info["calib"] = calib_info

        if annotations is not None:
            info['annos'] = annotations
            add_difficulty_to_annos(info)
        return info

    image_infos = []
    with ThreadPoolExecutor(num_worker) as pool:
        futures = [pool.submit(map_func, a) for a in image_ids]
        for f in tqdm(as_completed(futures), total=len(image_ids), desc=op_name):
            image_infos.append(f.result())

    return image_infos


def get_image_index_str(img_idx):
    return "{:06d}".format(img_idx)


def get_kitti_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True):
    img_idx_str = get_image_index_str(idx)
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path('training') / info_type / img_idx_str
    else:
        file_path = Path('testing') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_kitti_info_path(idx, prefix, 'image_2', '.png', training, relative_path, exist_check)


def get_label_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_kitti_info_path(idx, prefix, 'label_2', '.txt', training, relative_path, exist_check)


def get_velodyne_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_kitti_info_path(idx, prefix, 'velodyne', '.bin', training, relative_path, exist_check)


def get_calib_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt', training, relative_path, exist_check)


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def add_difficulty_to_annos(info):
    min_height = [40, 25, 25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [0, 1, 2]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [0.15, 0.3, 0.5]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool)
    hard_mask = np.ones((len(dims), ), dtype=np.bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos["difficulty"] = np.array(diff, np.int32)
    return diff


def add_difficulty_to_annos_v2(info):
    min_height = [40, 25, 25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [0, 1, 2]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [0.15, 0.3, 0.5]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = not ((occlusion > max_occlusion[0]) or
                     (height < min_height[0]) or
                     (truncation > max_trunc[0]))
    moderate_mask = not ((occlusion > max_occlusion[1]) or
                         (height < min_height[1]) or
                         (truncation > max_trunc[1]))
    hard_mask = not ((occlusion > max_occlusion[2]) or
                     (height < min_height[2]) or
                     (truncation > max_trunc[2]))
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos["difficulty"] = np.array(diff, np.int32)
    return diff


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]] for x in content]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def get_pseudo_label_anno():
    annotations = {}
    annotations.update({
        'name': np.array(['Car']),
        'truncated': np.array([0.0]),
        'occluded': np.array([0]),
        'alpha': np.array([0.0]),
        'bbox': np.array([[0.1, 0.1, 15.0, 15.0]]),
        'dimensions': np.array([[0.1, 0.1, 15.0, 15.0]]),
        'location': np.array([[0.1, 0.1, 15.0]]),
        'rotation_y': np.array([[0.1, 0.1, 15.0]])
    })
    return annotations


def get_start_result_anno():
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': [],
    })
    return annotations


def empty_result_anno():
    annotations = {}
    annotations.update({
        'name': np.array([]),
        'truncated': np.array([]),
        'occluded': np.array([]),
        'alpha': np.array([]),
        'bbox': np.zeros([0, 4]),
        'dimensions': np.zeros([0, 3]),
        'location': np.zeros([0, 3]),
        'rotation_y': np.array([]),
        'score': np.array([]),
    })
    return annotations


def get_label_annos(label_folder, image_ids=None):
    if image_ids is None:
        filepaths = Path(label_folder).glob('*.txt')
        prog = re.compile(r'^\d{6}.txt$')
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annos = []
    label_folder = Path(label_folder)
    for idx in image_ids:
        image_idx_str = get_image_index_str(idx)
        label_filename = label_folder / (image_idx_str + '.txt')
        anno = get_label_anno(label_filename)
        num_example = anno["name"].shape[0]
        anno["image_idx"] = np.array([idx] * num_example, dtype=np.int64)
        annos.append(anno)
    return annos
