import numpy as np


from spconv_lite.utils import non_max_suppression


def nms_gpu_cc(dets, nms_overlap_thresh, device_id=0):
    boxes_num = dets.shape[0]
    keep = np.zeros(boxes_num, dtype=np.int32)
    scores = dets[:, 4]
    order = scores.argsort()[::-1].astype(np.int32)
    sorted_dets = dets[order, :]
    num_out = non_max_suppression(sorted_dets, keep, nms_overlap_thresh,
                                  device_id)
    keep = keep[:num_out]
    return list(order[keep])
