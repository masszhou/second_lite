import cv2
import os
from time import time
import numpy as np
import fire
from pathlib import Path
from typing import Optional

from instance.detector import SecondDetector
from dataset.kitti import read_kitti_label
from dataset.kitti import KittiCalibrationParser


class Predict:
    def __init__(self):
        self.default_net_config_path = "./config/simple_second.yml"
        self.default_class_config_path = "./config/kitti_class.yml"
        self.default_ckpt_path = "./data/checkpoints/voxelnet-kitti-cls-lite-15450.tckpt"
        self.example_pcl_filepath = "/media/zzhou/data-KITTI/object/training/velodyne/000050.bin"
        self.example_pcl_folderpath = "/media/zzhou/data-KITTI/object/training/velodyne/"
        self.class_names =["car", "van", "pedestrian", "Cyclist"]

    def predict_pcl_files(self,
                          pcl_path: str = None,
                          image_root_path: str = None,
                          net_config_path: str = None,
                          class_config_path: str = None,
                          ckpt_path: str = None,
                          save: bool = False,
                          viewer_port: Optional[int] = None):
        """predict a single point cloud data from numpy array format
        expected shape=[n_points, xyzr=4]

        :param pcl_path: .bin file or folder with .bin files, saved numpy array data
        :param image_root_path:
        :param net_config_path: .yaml file, model setup
        :param class_config_path: dataset class configure
        :param ckpt_path: .tckpt file, trained weights
        :param show_gt: if display GT label
        :param save: save to video
        :param viewer_port:
        """

        if pcl_path is None:
            pcl_path = self.example_pcl_folderpath
        if net_config_path is None:
            net_config_path = self.default_net_config_path
        if class_config_path is None:
            class_config_path = self.default_class_config_path
        if ckpt_path is None:
            ckpt_path = self.default_ckpt_path

        if os.path.isdir(pcl_path):
            filenames = [Path(os.path.join(pcl_path, f))
                         for f in os.listdir(pcl_path)
                         if f.endswith(".bin")]
        else:
            filenames = [Path(pcl_path), ]
        filenames.sort()

        root_path = filenames[0].parent
        if image_root_path is None:
            image_root_path = Path(root_path) / "image_2"
        else:
            image_root_path = Path(image_root_path)

        if save:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out_stream = cv2.VideoWriter("./output.avi", fourcc, 10.0, (1000, 900))
        else:
            out_stream = None
        vis_img = np.zeros((900, 1000, 3), dtype=np.uint8)

        if viewer_port is not None:
            from utils.visualization import LidarViewer
            lidar_viewer = LidarViewer(viewer_port)
        else:
            lidar_viewer = None

        detector = SecondDetector(net_config_path, class_config_path, ckpt_path, detect_range=(-50, -50, 50, 50))

        for file_path in filenames:
            points = np.fromfile(file_path, dtype=np.float32, count=-1).reshape([-1, 4])
            print(file_path, points.shape)
            start = time()
            res = detector.predict_on_points(points)
            end = time()
            boxes_lidar = res[0]["box3d_lidar"].detach().cpu().numpy()
            scores = res[0]["scores"].detach().cpu().numpy()
            labels = res[0]["label_preds"].detach().cpu().numpy()

            if lidar_viewer is not None:
                lidar_viewer.load_points(points[:, :3], points[:, 3])
            # print("--------")
            # print(f">> filename: {file_path}")
            # print(f">> scores:   {scores}")
            # print(f">> labels:   {labels}")
            # print(f">> time:     {end-start} [s]")
            # print(">> boxes_lidar: ")
            # for each in boxes_lidar:
            #     print(each)  # (x, y, z, w, l, h, yaw) in velodyne coordinates
            bev = detector.visualize_bev(points, boxes_lidar, labels=labels)

            file_token = file_path.stem
            img = cv2.imread(str(image_root_path/(file_token + ".png")))
            img = cv2.resize(img, (1000, 300))
            vis_img[:300, :, :] = img
            vis_img[300:, :, :] = bev[::-1, :, :]

            # if show_gt:
            #     label_filepath = root_path / "label_2" / (file_token + ".txt")
            #     calib_filepath = root_path / "calib" / (file_token + ".txt")
            #     labels = read_kitti_label(label_filepath)
            #     calib = KittiCalibrationParser(calib_filepath)
            #     for each in labels:
            #         print(each)
            #         if each["name"] != "DontCare":
            #             # labels are in ref camera coord. thus here transform to lidar coord
            #             each["box"][:3] = calib.project_ref_to_velo(each["box"][:3])
            #             bev = detector.draw_box_in_bev(bev, each["box"].reshape([-1, 7]), labels)
            if save:
                out_stream.write(vis_img)

            cv2.imshow("bev",bev[::-1, :, :])
            cv2.imshow("img", vis_img)
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()
        if save:
            out_stream.release()


if __name__ == "__main__":
    fire.Fire(Predict)