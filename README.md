# 1. Introduction
simplified SECOND LiDAR object detection

my change:
* use own customized spconv_lite instead of spconv
* rewrite/refactor code
* trained with kitti
* trained with lyft 3d detection (not included in this repo)

# 2. Model Architecture
### 2.1 VFE module
### 2.2 middle conv3D module
### 2.3 RPN module

# 3. How to use

under root of this project
```
python -m script.predict_kitti predict_pcl_files --pcl_path="{kitti_root}/2011_09_26/2011_09_26_drive_0023_sync/velodyne_points/data/" --image_root_path="{kitti_root}/2011_09_26/2011_09_26_drive_0023_sync/image_02/data/" --save=True
```

# 4. Results on KITTI test set
training with 180° FOV labels, inference with 360° FOV
* green: car
* blue: van
* red: pedestrian
* yellow: cyclist

Results
* KITTI [2011-09-26-0005](https://youtu.be/p5ZlXYoMb5o)
* KITTI [2011 09 26 0023](https://youtu.be/fRAjDE7FdDQ)