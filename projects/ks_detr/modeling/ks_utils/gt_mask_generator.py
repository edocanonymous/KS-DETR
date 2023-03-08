import torch
import torch.nn.functional as F
import numpy as np

"""
https://discuss.pytorch.org/t/downsampling-tensors/16617/4
Upsample uses F.interpolate as suggested.
We can check the source to see what’s actually doing: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#interpolate 2.2k
The default mode ‘nearest’ doesn’t suit downscaling but ‘bilinear’ works fine (please correct me if I’m wrong!).
Otherwise, just use mode ‘area’ or adaptive_avg_pool2d directly. However, the latter seems a bit slower than ‘bilinear’.
"""

INTERPOLATE_MODE = 'bilinear'  # Never changing it back to nearest mode for 0, 1 downsamling interpolation


def estimate_sig_score_piecewise_linear(box_area_cuda):
    MIN_FG_SIGNIFICANCE = 0.6  # fg tokens have values larger or equal than MIN_FG_SIGNIFICANCE
    box_area = box_area_cuda.cpu().item()

    base_areas = np.array([32 ** 2, 64 ** 2, 128 ** 2, 256 ** 2, 512 ** 2])
    min_fg_significance = MIN_FG_SIGNIFICANCE
    sig_value_decay_per_piece = (1 - min_fg_significance) / (len(base_areas) - 1)

    # 0: base_areas < 32 ** 2, 1: 32 ** 2 < base_areas < 64 ** 2, 5: base_areas > 512 ** 2
    cnt = np.sum(box_area - base_areas > 0)  # count num of elements larger than box_area

    if cnt == 0:  # 0: base_areas < 32 ** 2,
        result = 1.0
    elif cnt == len(base_areas):
        result = min_fg_significance
    else:
        # 47 ** 2 -> cnt = 1
        l_v, r_v = base_areas[cnt - 1], base_areas[cnt]  # l_v, r_v = 32 ** 2, 64 ** 2
        # 1 - 0.1 *  [1 - (64 ** 2 - 47 ** 2) / (64 ** 2 - 32 ** 2)]
        result = 1 - sig_value_decay_per_piece * (cnt - (r_v - box_area) / (r_v - l_v))
    return result


def prepare_ksgt_targets(targets, padded_img_size, pad_fg_pixel=16, gt_fg_bg_mask_criterion=None):
    """
    Args:
        padded_img_size:
        gt_fg_bg_mask_criterion:
        pad_fg_pixel: should be stride / 2, to ensure we are conducting max_pooling when later we use
            bilinear intepolation mode in resizing.  For resnet, the stride of last layer is 32.

        targets: a list of dict, each dict contains the gt information of one image.
        each dict:
            'boxes' = {Tensor: 4} tensor([[0.5000, 0.5921, 1.0000, 0.8157],\n        [0.6338, 0.6836, 0.6812, 0.6058],\n        [0.5718, 0.2573, 0.4123, 0.5145],\n        [0.2712, 0.9666, 0.5423, 0.0669]], device='cuda:0')
            'labels' = {Tensor: 4} tensor([67, 47, 47, 60], device='cuda:0')
            'image_id' = {Tensor: 1} tensor([276037], device='cuda:0')
            'area' = {Tensor: 4} tensor([425523.2500, 215284.8281, 110670.8125,  18919.1719], device='cuda:0')
            'iscrowd' = {Tensor: 4} tensor([0, 0, 0, 0], device='cuda:0')
            'orig_size' = {Tensor: 2} tensor([640, 448], device='cuda:0') # img_h, img_w = tgt['orig_size'].unbind()
            'size' = {Tensor: 2} tensor([741, 704], device='cuda:0'), ([int(h), int(w)])
    Returns:
        torch.float()
           ksgt_targets_raw:  # cannot be used as gt.
               dict(
                    fg_gt=fg_gt,  # B, H, W
                )
    Due to random crop in the transpose,
    original box are is 1804, but in the cropped image, it occupies the whole image.
     True, 1804  < 9216, tensor([[0.5000, 0.5000, 1.0000, 1.0000],
        [0.5000, 0.5000, 1.0000, 1.0000]], device='cuda:0')
    """
    if gt_fg_bg_mask_criterion is None:
        gt_fg_bg_mask_criterion = 'Fg1Bg0'

    batch_size = len(targets)
    mask_size = (batch_size,) + padded_img_size

    fg_gt = torch.zeros(mask_size).to(targets[0]['boxes'].device).float()  # H, W

    for k, img_target in enumerate(targets):
        box_unnormalized = img_target['box_unnormalized'].type(torch.int64)
        num_box = len(img_target['boxes'])
        if num_box > 0:
            # Extend the fg regions
            if pad_fg_pixel > 0:
                input_img_size = img_target['image_size']  # input image size
                h, w = input_img_size[0].item(), input_img_size[1].item()
                offset = torch.tensor([-pad_fg_pixel, -pad_fg_pixel, pad_fg_pixel, pad_fg_pixel],
                                      dtype=torch.int32, device=box_unnormalized.device
                                      ).unsqueeze(dim=0).repeat(num_box, 1)
                box_unnormalized += offset
                box_unnormalized[:, 0::2].clamp_(min=0, max=w)  # w: x
                box_unnormalized[:, 1::2].clamp_(min=0, max=h)  # h: y

            for kk, box in enumerate(box_unnormalized):
                x1, y1, x2, y2 = box
                box_area = (x2 - x1) * (y2 - y1)

                if gt_fg_bg_mask_criterion == 'Fg1Bg0':
                    fg_gt[k, y1:y2, x1:x2] = 1.0  # foreground objects
                elif gt_fg_bg_mask_criterion == 'instance_mask':
                    instance_mask = img_target['masks'][kk]
                    padding_bottom, padding_right = padded_img_size[0] - instance_mask.shape[0], \
                                                    padded_img_size[1] - instance_mask.shape[1]
                    m = torch.nn.ZeroPad2d((0, padding_right, 0, padding_bottom))
                    instance_mask_padded = m(instance_mask.float().unsqueeze(0)).bool().squeeze(0)

                    fg_gt[k][instance_mask_padded] = 1.0

                elif gt_fg_bg_mask_criterion == 'significance_value':
                    # soft label for small object so that smaller objects has large value (e.g., 1 vs. 0.9)
                    #  than relative large small objects

                    significant_score = estimate_sig_score_piecewise_linear(box_area)

                    # Use the max significant_score if overlap exists
                    fg_gt[k, y1:y2, x1:x2] = torch.max(
                        fg_gt[k, y1:y2, x1:x2],
                        fg_gt.new_tensor(significant_score)
                    )
                else:
                    raise NotImplementedError

    ksgt_targets_raw = dict(fg_gt=fg_gt.float(), )

    return ksgt_targets_raw


def resize_ksgt_target(ksgt_targets, feat_map_size,
                       interpolate_mode=INTERPOLATE_MODE  # bilinear
                       ):
    """
    F.interpolate default mode is 'nearest'

    Args:
        interpolate_mode:
        ksgt_targets:
        feat_map_size:

    Returns: (N, B), float()

    """
    if torch.is_tensor(feat_map_size):
        output_size = tuple(feat_map_size.cpu().numpy())
    else:
        output_size = tuple(feat_map_size)

    ksgt_targets_resized = {}
    for k, gt in ksgt_targets.items():
        gt_new = F.interpolate(gt[None].float(), size=output_size, mode=interpolate_mode, align_corners=True)[0]

        ksgt_targets_resized[k] = gt_new.bool().float().flatten(1).permute(1, 0)  # float -> long  .long()

    return ksgt_targets_resized  # torch.Size([600, 2]), (N, B)


class TokenGTMaskGenerator:
    def __init__(self,
                 gt_fg_bg_mask_criterion,
                 pad_fg_pixel=None,
                 ):
        self.gt_fg_bg_mask_criterion = gt_fg_bg_mask_criterion

        self.pad_fg_pixel = pad_fg_pixel

    def get_gt_raw(self, targets, padded_img_size):
        """
        Args:
            padded_img_size:
            targets: a list of dict
        Returns:
        """
        ksgt_targets_raw = prepare_ksgt_targets(
            targets=targets, pad_fg_pixel=self.pad_fg_pixel, padded_img_size=padded_img_size,
            gt_fg_bg_mask_criterion=self.gt_fg_bg_mask_criterion)
        return ksgt_targets_raw

    @staticmethod
    def resize_gt_mask(ksgt_targets, feat_map_size):
        return resize_ksgt_target(
            ksgt_targets=ksgt_targets,
            feat_map_size=feat_map_size,
        )
