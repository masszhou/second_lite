import torch
from torch import stack as tstack
import numpy as np

from utils.torchplus.tools import torch_to_np_dtype
from utils.non_max_suppression import nms_gpu_cc
from utils.non_max_suppression import rotate_nms_cc


def second_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for VoxelNet
    similar to SSD regression encode
    see documents: voxel_feature_encoding.md

    :param boxes: ([N, 7] Tensor), normal boxes: x, y, z, l, w, h, r
    :param anchors: ([N, 7] Tensor)
    :param encode_angle_to_vector: if encode angle vector
    :param smooth_dim: the way to encode l,w,h

    :return one tensor feature for all voxels
    """
    box_ndim = anchors.shape[-1]
    cas, cgs = [], []
    if box_ndim > 7:
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg, *cgs = torch.split(boxes, 1, dim=-1)
    else:
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg = torch.split(boxes, 1, dim=-1)

    diagonal = torch.sqrt(la ** 2 + wa ** 2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha
    cts = [g - a for g, a in zip(cgs, cas)]
    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
        ht = hg / ha - 1
    else:
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
    if encode_angle_to_vector:
        rgx = torch.cos(rg)
        rgy = torch.sin(rg)
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return torch.cat([xt, yt, zt, wt, lt, ht, rtx, rty, *cts], dim=-1)
    else:
        rt = rg - ra
        return torch.cat([xt, yt, zt, wt, lt, ht, rt, *cts], dim=-1)


def second_box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box decode for VoxelNet in lidar
    similar to SSD regression decode
    see documents: voxel_feature_encoding.md

    :param box_encodings: ([N, 7] Tensor), normal boxes: x, y, z, w, l, h, r
    :param anchors: ([N, 7] Tensor), anchors
    :param encode_angle_to_vector:
    :param smooth_dim:
    """
    box_ndim = anchors.shape[-1]
    cas, cts = [], []
    if box_ndim > 7:
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        if encode_angle_to_vector:
            xt, yt, zt, wt, lt, ht, rtx, rty, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)
    else:
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        if encode_angle_to_vector:
            xt, yt, zt, wt, lt, ht, rtx, rty = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)

    # za = za + ha / 2
    # xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)
    diagonal = torch.sqrt(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:
        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
    if encode_angle_to_vector:
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = torch.atan2(rgy, rgx)
    else:
        rg = rt + ra
    cgs = [t + a for t, a in zip(cts, cas)]
    return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2 ** ndim, ndim)
    return corners


def corner_to_standup_nd(boxes_corner):
    ndim = boxes_corner.shape[2]
    standup_boxes = []
    for i in range(ndim):
        standup_boxes.append(torch.min(boxes_corner[:, :, i], dim=1)[0])
    for i in range(ndim):
        standup_boxes.append(torch.max(boxes_corner[:, :, i], dim=1)[0])
    return torch.stack(standup_boxes, dim=1)


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    rot_mat_T = torch.stack(
        [tstack([rot_cos, -rot_sin]),
         tstack([rot_sin, rot_cos])])
    return torch.einsum('aij,jka->aik', (points, rot_mat_T))


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners


def limit_period(val, offset=0.5, period = np.pi):
    return val - torch.floor(val / period + offset) * period


# def multiclass_nms(nms_func,
#                    boxes,
#                    scores,
#                    num_class,
#                    pre_max_size=None,
#                    post_max_size=None,
#                    score_thresh=0.0,
#                    iou_threshold=0.5):
#     # only output [selected] * num_class, please slice by your self
#     selected_per_class = []
#     assert len(boxes.shape) == 3, "bbox must have shape [N, num_cls, 7]"
#     assert len(scores.shape) == 2, "score must have shape [N, num_cls]"
#     num_class = scores.shape[1]
#     if not (boxes.shape[1] == scores.shape[1] or boxes.shape[1] == 1):
#         raise ValueError('second dimension of boxes must be either 1 or equal '
#                          'to the second dimension of scores')
#     num_boxes = boxes.shape[0]
#     num_scores = scores.shape[0]
#     num_classes = scores.shape[1]
#     boxes_ids = (range(num_classes)
#                  if boxes.shape[1] > 1 else [0] * num_classes)
#     for class_idx, boxes_idx in zip(range(num_classes), boxes_ids):
#         # for class_idx in range(1, num_class):
#         class_scores = scores[:, class_idx]
#         class_boxes = boxes[:, boxes_idx]
#         if score_thresh > 0.0:
#             class_scores_keep = torch.nonzero(class_scores >= score_thresh)
#             if class_scores_keep.shape[0] != 0:
#                 class_scores_keep = class_scores_keep[:, 0]
#             else:
#                 selected_per_class.append(None)
#                 continue
#             class_scores = class_scores[class_scores_keep]
#         if class_scores.shape[0] != 0:
#             if score_thresh > 0.0:
#                 class_boxes = class_boxes[class_scores_keep]
#             keep = nms_func(class_boxes, class_scores, pre_max_size,
#                             post_max_size, iou_threshold)
#             if keep.shape[0] != 0:
#                 if score_thresh > 0.0:
#                     selected_per_class.append(class_scores_keep[keep])
#                 else:
#                     selected_per_class.append(keep)
#             else:
#                 selected_per_class.append(None)
#         else:
#             selected_per_class.append(None)
#     return selected_per_class
#
#
def nms(bboxes,
        scores,
        pre_max_size=None,
        post_max_size=None,
        iou_threshold=0.5):
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = torch.topk(scores, k=pre_max_size)
        bboxes = bboxes[indices]
    dets = torch.cat([bboxes, scores.unsqueeze(-1)], dim=1)
    dets_np = dets.data.cpu().numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(nms_gpu_cc(dets_np, iou_threshold), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return torch.zeros([0]).long().to(bboxes.device)
    if pre_max_size is not None:
        keep = torch.from_numpy(keep).long().to(bboxes.device)
        return indices[keep]
    else:
        return torch.from_numpy(keep).long().to(bboxes.device)


def rotate_nms(rbboxes,
               scores,
               pre_max_size=None,
               post_max_size=None,
               iou_threshold=0.5):
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = torch.topk(scores, k=pre_max_size)
        rbboxes = rbboxes[indices]
    dets = torch.cat([rbboxes, scores.unsqueeze(-1)], dim=1)
    dets_np = dets.data.cpu().numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(rotate_nms_cc(dets_np, iou_threshold), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return torch.zeros([0]).long().to(rbboxes.device)
    if pre_max_size is not None:
        keep = torch.from_numpy(keep).long().to(rbboxes.device)
        return indices[keep]
    else:
        return torch.from_numpy(keep).long().to(rbboxes.device)