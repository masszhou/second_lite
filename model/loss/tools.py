import torch
import utils.torchplus as torchplus


def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                sin_error_factor=1.0,
                box_code_size=7,
                num_direction_bins=2):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = torchplus.nn.one_hot(cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        # reg_tg_rot = box_torch_ops.limit_period(
        #     reg_targets[..., 6:7], 0.5, 2 * np.pi / num_direction_bins)
        box_preds, reg_targets = add_sin_difference(box_preds,
                                                    reg_targets,
                                                    box_preds[..., 6:7],
                                                    reg_targets[..., 6:7],
                                                    sin_error_factor)

    loc_losses = loc_loss_ftor(box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    return loc_losses, cls_losses


def add_sin_difference(boxes1, boxes2, boxes1_rot, boxes2_rot, factor=1.0):
    if factor != 1.0:
        boxes1_rot = factor * boxes1_rot
        boxes2_rot = factor * boxes2_rot
    rad_pred_encoding = torch.sin(boxes1_rot) * torch.cos(boxes2_rot)
    rad_tg_encoding = torch.cos(boxes1_rot) * torch.sin(boxes2_rot)
    boxes1 = torch.cat([boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
    boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]], dim=-1)
    return boxes1, boxes2
