import copy

import cv2
import os

import smrc.utils
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from TCFormer.tcformer_module.tcformer_utils import *
import math
from nms import ProposalProcess

# areaRngSGDT = [0 ** 2, 32 ** 2, 96 ** 2, 1e5 ** 2]  # defined in the original image space.
areaRngSGDT = [0 ** 2, 32 ** 2, 96 ** 2, 128 ** 2, 256 ** 2, 1e5 ** 2]  # Here we define in the input image space.


def is_small_object(obj_area):
    if obj_area < areaRngSGDT[3]:  # small + medium
        return True
    else:
        return False


def unnormalize_box(box_normalized, input_img_size):
    """
            # sgdt_output['feat_map_size'] = torch.tensor([h, w])
        # feat_map_size = sgdt_output['feat_map_size']  # [h, w] tensor([23, 31], device='cuda:0')

                # aspect_ratio = padded_img_size / feat_map_size  # padded input image / padded feat map

            # box coordinates are defined in input_img space, here down scale the box to fit the size of
            # feature map
            # box_unnormalized = box_unnormalized / aspect_ratio.unsqueeze(dim=0).repeat(num_box, 2)
            # extend the boundary 0.5 pixel on the feature map to encourage the margin of box to be classified
            # as fg token.
    Args:
        box_normalized:
        input_img_size:

    Returns:

    """
    h, w = input_img_size[0].item(), input_img_size[1].item()
    ratio = torch.tensor([w, h, w, h], dtype=torch.float32, device=box_normalized.device)

    num_box = len(box_normalized)
    if num_box > 0:
        box_unnormalized = box_normalized * ratio.unsqueeze(dim=0).repeat(num_box, 1)

        #  cx, cy, w, h to x1, y1, x2, y2, but slightly expand the box: floor for x1, y1;
        #  ceil for x2, y2 to cover all the object region.
        box_unnormalized = torch.floor(torch.stack(
            [box_unnormalized[:, 0] - box_unnormalized[:, 2] / 2,
             box_unnormalized[:, 1] - box_unnormalized[:, 3] / 2,
             box_unnormalized[:, 0] + box_unnormalized[:, 2] / 2 + 1,
             box_unnormalized[:, 1] + box_unnormalized[:, 3] / 2 + 1],  #
            dim=-1)).int()
        # print(f'before clamp, box_unnormalized = {box_unnormalized}')
        # box_unnormalized[:, 0::2] = torch.clamp(box_unnormalized[:, 0::2], min=0, max=w)
        # box_unnormalized[:, 1::2] = torch.clamp(box_unnormalized[:, 1::2], min=0, max=h)
        box_unnormalized[:, 0::2].clamp_(min=0, max=w)  # w: x
        box_unnormalized[:, 1::2].clamp_(min=0, max=h)  # h: y

    else:
        box_unnormalized = box_normalized
    # print(f'after clamp box_unnormalized = {box_unnormalized}')
    return box_unnormalized


# class TokenScoringTarget:
#     def __init__(self, max_box_area=1024 ** 2):
#         self.max_box_area = max_box_area


MAX_BOX_AREA = 512 ** 2  # (256 + 64) ** 2
MIN_FG_SIGNIFICANCE = 0.6  # only used in this file


def estimate_significant_score(box_area):
    # smaller area has larger score

    # # clip the box_area if it is larger than the defined max_box_area
    # box_area = min(MAX_BOX_AREA, box_area)
    # significant_score = (MAX_BOX_AREA - box_area) / MAX_BOX_AREA

    box_area = min(MAX_BOX_AREA, box_area)
    significant_score = (math.log(MAX_BOX_AREA) - math.log(box_area)) / math.log(MAX_BOX_AREA)
    # shift the score to the range of [0.5, 1], so that fg tokens has value >= 0.5
    significant_score = 0.5 + significant_score / 2.0

    return significant_score


def estimate_sig_score_piecewise_linear(box_area_cuda):
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


def prepare_sgdt_proposal_significant_value(
        proposals, pad_fg_pixel=16, token_scoring_gt_criterion=None,
        min_fg_score=0.0, min_split_score=0.0):
    """

    Args:
        token_scoring_gt_criterion:
        pad_fg_pixel: should stride / 2, to ensure we are conducting max_pooling when later we use
            bilinear intepolation mode in resizing.  For resnet the last layer, stride is 32.

        proposals: a list of dict, each dict contains the gt information of one image.
        each dict:
            'boxes' = {Tensor: 4} tensor([[0.5000, 0.5921, 1.0000, 0.8157],\n        [0.6338, 0.6836, 0.6812, 0.6058],\n        [0.5718, 0.2573, 0.4123, 0.5145],\n        [0.2712, 0.9666, 0.5423, 0.0669]], device='cuda:0')
            'labels' = {Tensor: 4} tensor([67, 47, 47, 60], device='cuda:0')
            'image_id' = {Tensor: 1} tensor([276037], device='cuda:0')
            'area' = {Tensor: 4} tensor([425523.2500, 215284.8281, 110670.8125,  18919.1719], device='cuda:0')
            'iscrowd' = {Tensor: 4} tensor([0, 0, 0, 0], device='cuda:0')
            'orig_size' = {Tensor: 2} tensor([640, 448], device='cuda:0') # img_h, img_w = tgt['orig_size'].unbind()
            'size' = {Tensor: 2} tensor([741, 704], device='cuda:0'), ([int(h), int(w)])
    Returns: torch.float()
           sgdt_targets_raw:  # cannot be used as gt.
               dict(
                    fg_gt=fg_gt,  # B, H, W
                    scale_gt=scale_gt  #  B, H, W
                )
    Due to random crop in the transpose,
    original box are is 1804, but in the cropped image, it occupies the whole image.
     True, 1804  < 9216, tensor([[0.5000, 0.5000, 1.0000, 1.0000],
        [0.5000, 0.5000, 1.0000, 1.0000]], device='cuda:0')
    """
    if token_scoring_gt_criterion is None:
        token_scoring_gt_criterion = 'significance_value'

    # B, H, W
    padded_img_size = proposals[0]['padded_img_size']  # (736, 981)
    batch_size = len(proposals)
    mask_size = (batch_size,) + tuple(padded_img_size.cpu().numpy())

    # We must not use scale_gt = fg_gt = torch.zeros(), otherwise, they will share the same variable.
    fg_gt = torch.zeros(mask_size).to(proposals[0]['size'].device).float()  # H, W  TODO
    scale_gt = torch.zeros(mask_size).to(proposals[0]['size'].device).float()

    # padded_img_area = torch.prod(padded_img_size)
    for k, img_target in enumerate(proposals):
        if token_scoring_gt_criterion == 'fake_all_tokens_are_fg':
            scale_gt = torch.ones_like(scale_gt)
            fg_gt = torch.ones_like(fg_gt)
            continue

        # 0 means bg, 1, fg. -1 means padding position.
        box_unnormalized = unnormalize_box(box_normalized=img_target['boxes'],
                                           input_img_size=img_target['size'])

        num_box = len(img_target['boxes'])
        if num_box > 0:  # tensor([32.0000, 31.6452], device='cuda:0')
            # ------------------------- Extend the fg regions
            if pad_fg_pixel > 0:
                input_img_size = img_target['size']
                h, w = input_img_size[0].item(), input_img_size[1].item()
                offset = torch.tensor([-pad_fg_pixel, -pad_fg_pixel, pad_fg_pixel, pad_fg_pixel],
                                      dtype=torch.int32, device=box_unnormalized.device
                                      ).unsqueeze(dim=0).repeat(num_box, 1)
                box_unnormalized += offset
                box_unnormalized[:, 0::2].clamp_(min=0, max=w)  # w: x
                box_unnormalized[:, 1::2].clamp_(min=0, max=h)  # h: y

            # -------------------------------- Generate the gt mask ('original_area', 'area')
            # Using the area of the box in the original img, instead of the input image, which will be changed in
            # self._transforms is a good choice.
            # But original area has more box then the final box, and we do not know the correspondence
            # for the box list with different number of boxes, so we cannot use the original area.
            # assert len(box_unnormalized) == len(img_target['area'])
            for kk, box in enumerate(box_unnormalized):
                # we use the recalculated box_are instead of the saved area because we may use the area of proposal
                # in that case, we cannot pre-save the area.
                x1, y1, x2, y2 = box
                box_area = (x2 - x1) * (y2 - y1)
                assert box_area >= 0
                # fg_gt[k, y1:y2, x1:x2] = True  # foreground objects

                # soft label for small object so that smaller objects has large value (e.g., 1 vs. 0.9)
                #  than relative large small objects

                if 'scores' in img_target and img_target['scores'][kk] < min_fg_score:
                    pass  # if score is too lower, ignore this for remove judgement.
                else:
                    fg_gt[k, y1:y2, x1:x2] = 1.0  # foreground objects

                if 'scores' in img_target and img_target['scores'][kk] < min_split_score:
                    pass  # skip generating the significant value for this box.
                else:
                    if token_scoring_gt_criterion.find('significance_value') > -1:
                        # significance_value_bg_w_priority

                        # significant_score = estimate_significant_score(box_area)
                        significant_score = estimate_sig_score_piecewise_linear(box_area)

                        if token_scoring_gt_criterion == 'significance_value_inverse_fg':
                            # inverse the significance of fg objects, so that larger has higher significance value.
                            # 1 -> MIN_FG_SIGNIFICANCE, MIN_FG_SIGNIFICANCE -> 1
                            significant_score = 1 - significant_score + MIN_FG_SIGNIFICANCE

                            fg_loc = scale_gt[k, y1:y2, x1:x2] > 0
                            bg_loc = scale_gt[k, y1:y2, x1:x2] <= 0
                            # Use the min significant_score if overlap exists so that the clutter regions has
                            # low priority to be sampled (only for debugging)
                            scale_gt[k, y1:y2, x1:x2][fg_loc] = torch.min(
                                scale_gt[k, y1:y2, x1:x2][fg_loc],
                                scale_gt.new_tensor(significant_score)
                            )
                            # for bg locations, just update to significant_score
                            scale_gt[k, y1:y2, x1:x2][bg_loc] = scale_gt.new_tensor(significant_score)
                        else:
                            # Use the max significant_score if overlap exists
                            scale_gt[k, y1:y2, x1:x2] = torch.max(
                                scale_gt[k, y1:y2, x1:x2],
                                scale_gt.new_tensor(significant_score)
                            )
                    elif token_scoring_gt_criterion == 'fg_scale_class_all_fg':
                        scale_gt[k, y1:y2, x1:x2] = 1.0
                    elif token_scoring_gt_criterion == 'fg_scale_class_small_medium_random':
                        # TODO: soft label for small object so that smaller objects has large value (e.g., 1 vs. 0.9)
                        #  than relative large small objects
                        if is_small_object(box_area):  # small object or not
                            scale_gt[k, y1:y2, x1:x2] = 1.0
                    else:
                        raise NotImplementedError
                    # else:
                    #     print(f' False, {int(box_area)}  > {areaRngSGDT[2]}')

    # if token_scoring_gt_criterion.find('significance_value') > -1:
    #     fg_gt = torch.where(scale_gt.float() > 0, 1.0, 0.0)
    # # else:
    # #     raise NotImplementedError

    sgdt_targets_raw = dict(
        fg_gt=fg_gt.float(),  # B, H, W
        scale_gt=scale_gt.float()  # B, H, W
    )

    return sgdt_targets_raw


def prepare_sgdt_targets(targets, pad_fg_pixel=16, token_scoring_gt_criterion=None,
                         min_fg_score=0.1, min_split_score=0.1):  # -float('inf')
    """

    Args:
        token_scoring_gt_criterion:
        pad_fg_pixel: should stride / 2, to ensure we are conducting max_pooling when later we use
            bilinear intepolation mode in resizing.  For resnet the last layer, stride is 32.

        targets: a list of dict, each dict contains the gt information of one image.
        each dict:
            'boxes' = {Tensor: 4} tensor([[0.5000, 0.5921, 1.0000, 0.8157],\n        [0.6338, 0.6836, 0.6812, 0.6058],\n        [0.5718, 0.2573, 0.4123, 0.5145],\n        [0.2712, 0.9666, 0.5423, 0.0669]], device='cuda:0')
            'labels' = {Tensor: 4} tensor([67, 47, 47, 60], device='cuda:0')
            'image_id' = {Tensor: 1} tensor([276037], device='cuda:0')
            'area' = {Tensor: 4} tensor([425523.2500, 215284.8281, 110670.8125,  18919.1719], device='cuda:0')
            'iscrowd' = {Tensor: 4} tensor([0, 0, 0, 0], device='cuda:0')
            'orig_size' = {Tensor: 2} tensor([640, 448], device='cuda:0') # img_h, img_w = tgt['orig_size'].unbind()
            'size' = {Tensor: 2} tensor([741, 704], device='cuda:0'), ([int(h), int(w)])
    Returns: torch.float()
           sgdt_targets_raw:  # cannot be used as gt.
               dict(
                    fg_gt=fg_gt,  # B, H, W
                    scale_gt=scale_gt  #  B, H, W
                )
    Due to random crop in the transpose,
    original box are is 1804, but in the cropped image, it occupies the whole image.
     True, 1804  < 9216, tensor([[0.5000, 0.5000, 1.0000, 1.0000],
        [0.5000, 0.5000, 1.0000, 1.0000]], device='cuda:0')
    """
    if token_scoring_gt_criterion is None:
        token_scoring_gt_criterion = 'significance_value'

    # B, H, W
    padded_img_size = targets[0]['padded_img_size']  # (736, 981)
    batch_size = len(targets)
    mask_size = (batch_size,) + tuple(padded_img_size.cpu().numpy())

    # We must not use scale_gt = fg_gt = torch.zeros(), otherwise, they will share the same variable.
    fg_gt = torch.zeros(mask_size).to(targets[0]['size'].device).float()  # H, W  TODO
    scale_gt = torch.zeros(mask_size).to(targets[0]['size'].device).float()

    # padded_img_area = torch.prod(padded_img_size)
    for k, img_target in enumerate(targets):
        if token_scoring_gt_criterion == 'fake_all_tokens_are_fg':
            scale_gt = torch.ones_like(scale_gt)
            fg_gt = torch.ones_like(fg_gt)
            continue

        # 0 means bg, 1, fg. -1 means padding position.
        box_unnormalized = unnormalize_box(box_normalized=img_target['boxes'],
                                           input_img_size=img_target['size'])

        num_box = len(img_target['boxes'])
        if num_box > 0:  # tensor([32.0000, 31.6452], device='cuda:0')
            # ------------------------- Extend the fg regions
            if pad_fg_pixel > 0:
                input_img_size = img_target['size']
                h, w = input_img_size[0].item(), input_img_size[1].item()
                offset = torch.tensor([-pad_fg_pixel, -pad_fg_pixel, pad_fg_pixel, pad_fg_pixel],
                                      dtype=torch.int32, device=box_unnormalized.device
                                      ).unsqueeze(dim=0).repeat(num_box, 1)
                box_unnormalized += offset
                box_unnormalized[:, 0::2].clamp_(min=0, max=w)  # w: x
                box_unnormalized[:, 1::2].clamp_(min=0, max=h)  # h: y

            # -------------------------------- Generate the gt mask ('original_area', 'area')
            # Using the area of the box in the original img, instead of the input image, which will be changed in
            # self._transforms is a good choice.
            # But original area has more box then the final box, and we do not know the correspondence
            # for the box list with different number of boxes, so we cannot use the original area.
            # assert len(box_unnormalized) == len(img_target['area'])
            for kk, box in enumerate(box_unnormalized):
                # we use the recalculated box_are instead of the saved area because we may use the area of proposal
                # in that case, we cannot pre-save the area.
                x1, y1, x2, y2 = box
                box_area = (x2 - x1) * (y2 - y1)
                assert box_area >= 0
                # fg_gt[k, y1:y2, x1:x2] = True  # foreground objects

                # soft label for small object so that smaller objects has large value (e.g., 1 vs. 0.9)
                #  than relative large small objects

                if 'scores' in img_target and img_target['scores'][kk] < min_fg_score:
                    pass  # if score is too lower, ignore this for remove judgement.
                else:
                    fg_gt[k, y1:y2, x1:x2] = 1.0  # foreground objects

                if 'scores' in img_target and img_target['scores'][kk] < min_split_score:
                    pass  # skip generating the significant value for this box.
                else:
                    if token_scoring_gt_criterion.find('significance_value') > -1:
                        # significance_value_bg_w_priority

                        # significant_score = estimate_significant_score(box_area)
                        significant_score = estimate_sig_score_piecewise_linear(box_area)

                        if token_scoring_gt_criterion == 'significance_value_inverse_fg':
                            # inverse the significance of fg objects, so that larger has higher significance value.
                            # 1 -> MIN_FG_SIGNIFICANCE, MIN_FG_SIGNIFICANCE -> 1
                            significant_score = 1 - significant_score + MIN_FG_SIGNIFICANCE

                            fg_loc = scale_gt[k, y1:y2, x1:x2] > 0
                            bg_loc = scale_gt[k, y1:y2, x1:x2] <= 0
                            # Use the min significant_score if overlap exists so that the clutter regions has
                            # low priority to be sampled (only for debugging)
                            scale_gt[k, y1:y2, x1:x2][fg_loc] = torch.min(
                                scale_gt[k, y1:y2, x1:x2][fg_loc],
                                scale_gt.new_tensor(significant_score)
                            )
                            # for bg locations, just update to significant_score
                            scale_gt[k, y1:y2, x1:x2][bg_loc] = scale_gt.new_tensor(significant_score)
                        else:
                            # Use the max significant_score if overlap exists
                            scale_gt[k, y1:y2, x1:x2] = torch.max(
                                scale_gt[k, y1:y2, x1:x2],
                                scale_gt.new_tensor(significant_score)
                            )
                    elif token_scoring_gt_criterion == 'fg_scale_class_all_fg':
                        scale_gt[k, y1:y2, x1:x2] = 1.0
                    elif token_scoring_gt_criterion == 'fg_scale_class_small_medium_random':
                        # TODO: soft label for small object so that smaller objects has large value (e.g., 1 vs. 0.9)
                        #  than relative large small objects
                        if is_small_object(box_area):  # small object or not
                            scale_gt[k, y1:y2, x1:x2] = 1.0
                    else:
                        raise NotImplementedError
                    # else:
                    #     print(f' False, {int(box_area)}  > {areaRngSGDT[2]}')

    # if token_scoring_gt_criterion.find('significance_value') > -1:
    #     fg_gt = torch.where(scale_gt.float() > 0, 1.0, 0.0)
    # # else:
    # #     raise NotImplementedError

    sgdt_targets_raw = dict(
        fg_gt=fg_gt.float(),  # B, H, W
        scale_gt=scale_gt.float()  # B, H, W
    )

    return sgdt_targets_raw


# def interpolate_for_max_pool()

def resize_sgdt_target_v0_deprecated(sgdt_targets, feat_map_size,
                                     feat_map_mask=None,
                                     interpolate_mode='bilinear'  # nearest
                                     ):
    """
    F.interpolate default mode is 'nearest'
    Args:
        interpolate_mode:
        sgdt_targets:
        feat_map_size:
        feat_map_mask: (B, H, W), bool, True means padded tokens (invalid, not be used in computation)

    Returns: float(), float()

    """
    assert interpolate_mode != 'nearest', 'nearest interpolation will cause round off, we need max pooling' \
                                          'operation here.'
    fg_gt, scale_gt = sgdt_targets['fg_gt'], sgdt_targets['scale_gt']
    # scale_gt = sgdt_targets['scale_gt']
    # B, H, W -> B, h, w (feature map size) size=x.shape[-2:]
    # feat_map_size = sgdt_output['feat_map_size']
    if torch.is_tensor(feat_map_size):
        output_size = tuple(feat_map_size.cpu().numpy())
    else:
        # if not isinstance(feat_map_size, (tuple, list)):
        output_size = tuple(feat_map_size)

    fg_gt_binary = True if fg_gt.unique().shape[0] == 2 else False
    scale_gt_binary = True if scale_gt.unique().shape[0] == 2 else False

    fg_gt = F.interpolate(fg_gt[None].float(), size=output_size, mode=interpolate_mode)[0]
    scale_gt = F.interpolate(scale_gt[None].float(),
                             size=output_size, mode=interpolate_mode)[0]  # torch.float32
    # fg_gt = F.interpolate(fg_gt[None].float(), size=output_size).to(torch.bool)[0]  # torch.Size([2, 23, 31])
    # scale_gt = F.interpolate(scale_gt[None].float(),
    #                          size=output_size, mode='nearest',).to(torch.bool)[0]

    # for binary value, we should conduct max_pooling operation to avoid round off error
    # That is, no fg grid should be marked as bg even if only a portion of the pixels are non-zeros.
    if fg_gt_binary: fg_gt = fg_gt.bool().float()
    if scale_gt_binary: scale_gt = scale_gt.bool().float()

    # # ======================== only for debugging, TODO: remove the following lines
    # # no need to do the following operation
    # if feat_map_mask is not None:
    #     ErrorFlag = False
    #     if fg_gt[feat_map_mask].sum() > 0:
    #         print(f'fg_gt[feat_map_mask].sum() = {fg_gt[feat_map_mask].sum()}')
    #         ErrorFlag = True
    #     if scale_gt[feat_map_mask].sum() > 0:
    #         print(f'fg_gt[feat_map_mask].sum() = {scale_gt[feat_map_mask].sum()}')
    #         ErrorFlag = True
    #     if ErrorFlag:
    #         raise ErrorFlag
    #
    #     # fg_gt[feat_map_mask] = False
    #     scale_gt[feat_map_mask] = False
    # ========================

    sgdt_targets = dict(  # B, H, W -> HxW, B, and then will be expanded to (N, B, Num_Class)
        fg_gt=fg_gt.flatten(1).permute(1, 0),  # float -> long  .long()
        scale_gt=scale_gt.flatten(1).permute(1, 0)  # float -> long  .long()
    )
    return sgdt_targets


def resize_sgdt_target(sgdt_targets, feat_map_size,
                       feat_map_mask=None,
                       interpolate_mode='bilinear'  # nearest
                       ):
    """
    F.interpolate default mode is 'nearest'
    Args:
        interpolate_mode:
        sgdt_targets:
        feat_map_size:
        feat_map_mask: (B, H, W), bool, True means padded tokens (invalid, not be used in computation)

    Returns: float(), float()

    """
    assert interpolate_mode != 'nearest', 'nearest interpolation will cause round off, we need max pooling' \
                                          'operation here.'

    if torch.is_tensor(feat_map_size):
        output_size = tuple(feat_map_size.cpu().numpy())
    else:
        # if not isinstance(feat_map_size, (tuple, list)):
        output_size = tuple(feat_map_size)

    sgdt_targets_resized = {}
    for k, gt in sgdt_targets.items():  # gt could be fg_gt, scale_gt, proposal_fg_gt, proposal_scale_gt
        gt_binary = True if gt.unique().shape[0] == 2 else False
        gt_new = F.interpolate(gt[None].float(), size=output_size, mode=interpolate_mode)[0]

        if gt_binary:
            gt_new = gt_new.bool().float()

        sgdt_targets_resized[k] = gt_new.flatten(1).permute(1, 0)  # float -> long  .long()

    return sgdt_targets_resized


class TokenScoringGTGenerator:
    def __init__(self,
                 token_scoring_gt_criterion,
                 pad_fg_pixel=None
                 ):
        self.token_scoring_gt_criterion = token_scoring_gt_criterion

        self.pad_fg_pixel = pad_fg_pixel

        self.sig_value_interpolate_mode = 'bilinear'  # nearest cause round off error
        # if token_scoring_gt_criterion == 'significance_value_bg_w_priority':
        #     # https://imagingsolution.blog.fc2.com/blog-entry-142.html
        #     self.sig_value_interpolate_mode = 'bilinear'

    def get_gt_raw(self, targets, **kwargs):
        """

        Args:
            targets: a list of dict

        Returns:

        """
        sgdt_targets_raw = prepare_sgdt_targets(
            targets=targets, pad_fg_pixel=self.pad_fg_pixel,
            token_scoring_gt_criterion=self.token_scoring_gt_criterion)

        if 'proposal_boxes' in targets[0]:
            # extract proposals as targets
            targets_proposal = extract_proposals_as_targets(targets)

            min_score = kwargs.pop('min_score', None)
            nms_thd = kwargs.pop('nms_thd', None)
            num_select = kwargs.pop('num_select', None)
            if min_score is not None or nms_thd is not None or num_select is not None:
                proposal_processor = ProposalProcess(
                    min_score=min_score,
                    nms_thd=nms_thd,
                    num_select=num_select)
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                targets_proposal = proposal_processor(
                    targets_proposal,
                    target_sizes=orig_target_sizes)

            # generate gt from proposals.
            sgdt_targets_proposal_raw = prepare_sgdt_proposal_significant_value(
                proposals=targets_proposal, pad_fg_pixel=self.pad_fg_pixel,
                token_scoring_gt_criterion=self.token_scoring_gt_criterion,
                min_fg_score=kwargs.pop('min_fg_score', 0.1),
                min_split_score=kwargs.pop('min_split_score', 0.1),
            )


            sgdt_targets_raw['proposal_fg_gt'] = sgdt_targets_proposal_raw['fg_gt']
            sgdt_targets_raw['proposal_scale_gt'] = sgdt_targets_proposal_raw['scale_gt']

        return sgdt_targets_raw

    def resize_sig_value_gt(self, sgdt_targets, feat_map_size):
        return resize_sgdt_target(
            sgdt_targets=sgdt_targets,
            feat_map_size=feat_map_size,
            interpolate_mode=self.sig_value_interpolate_mode
        )


def classify_predicted_remove_tokens(tokens_to_discard, fg_gt):
    """
    This function can also be used for split token classification, in that case,
    tokens_to_discard = tokens to not split, fg_gt = split_gt.
    Args:
        tokens_to_discard:
        fg_gt:

    Returns:

    """
    assert tokens_to_discard.shape == fg_gt.shape

    bg_tokens_correct = tokens_to_discard.float() * (1 - fg_gt.float())
    bg_tokens_missed = (1 - tokens_to_discard.float()) * (1 - fg_gt.float())
    # false remove
    fg_tokens_missed = tokens_to_discard.float() * fg_gt.float()

    return bg_tokens_correct, fg_tokens_missed, bg_tokens_missed


def classify_predicted_split_tokens(tokens_to_split, split_gt):
    """
    This function can also be used for split token classification, in that case,
    tokens_to_discard = tokens to not split, fg_gt = split_gt.
    Args:
        tokens_to_split:
        split_gt:

    Returns:

    """
    assert tokens_to_split.shape == split_gt.shape
    split_gt_float = split_gt.bool().float()  # change all non-zero locations to 1.
    split_tokens_correct = tokens_to_split.float() * split_gt_float
    split_tokens_missed = (1 - tokens_to_split.float()) * split_gt_float

    # keep_tokens_missed
    false_split_tokens = tokens_to_split.float() * (1 - split_gt_float)

    return split_tokens_correct, false_split_tokens, split_tokens_missed


from tti.tti_conf import VISUALIZATION_DIR


class VisualizeToken:
    """

        Args:
            targets: a tuple, each item for one input image.
                'boxes' = {Tensor: 2} tensor([[0.5661, 0.4576, 0.8191, 0.8889],\n        [0.2402, 0.4000, 0.1506, 0.1176]])
                'labels' = {Tensor: 2} tensor([ 1, 40])
                'image_id' = {Tensor: 1} tensor([270749])
                'area' = {Tensor: 2} tensor([347517.0625,   8449.1152])
                'iscrowd' = {Tensor: 2} tensor([0, 0])
                'orig_size' = {Tensor: 2} tensor([640, 476])
                'size' = {Tensor: 2} tensor([785, 608])

                'input_imgs' = nly saved in the last example, torch.Size([2, 3, 1105, 736]) with padded element

                # torch.Size([3, 785, 608])
            sgdt_target_raw: a dict,
                        fg_gt=fg_gt,  # B, H, W
                        scale_gt=scale_gt  #  B, H, W

            # sgdt_targets = dict(  # B, H, W -> HxW, B, and then will be expanded to (N, B, Num_Class)
            #         fg_gt=fg_gt.flatten(1).permute(1, 0), # N, B
            #         scale_gt=scale_gt.flatten(1).permute(1, 0)   #  N, B
            # )

            sgdt_output:  a dict,
                         dict(
                            adapted_pos=adapted_pos,  # torch.Size([805, 2, 256]), N, B, C
                            fg_score_logit=fg_score_logit,  # B, N; torch.Size([2, 630])
                            small_scale_score_logit=small_scale_score_logit,
                        )
                # fg_score, scale_score  # N, B, C, where C is the number of classes,
                # e.g., torch.Size([650, 2, 2]), torch.Size([630, 2, 3]);
                each score: probability not logits (sum to 1 for each prediction of one token)


            token_dict = {
                'x': x,
                'token_num': N,
                'map_size': [H, W],
                'init_grid_size': [H, W],
                'idx_token': idx_token,
                'agg_weight': agg_weight
            }
        Returns:

        """

    def __init__(self, targets, sgdt_target_raw, sgdt_targets, sgdt_output, out_dir=None):
        self.targets = targets
        self.sgdt_target_raw = sgdt_target_raw
        self.sgdt_output = sgdt_output
        self.sgdt_targets = sgdt_targets

        if out_dir is None:
            out_dir = VISUALIZATION_DIR
        self.out_dir = out_dir

    def visualize_token_adaption(self, sub_dir=None):
        # self.visualize_fg_scale_gt()

        # self.visualize_prediction()
        # self.visualize_l1_loss_prediction()
        #
        self.visualize_fg_scale_prediction(
            light_view=True, only_padded_image=True, disable_remove_view=True, sub_dir=sub_dir)  #

    def visualize_split_token(self, sub_dir=None):
        # self.visualize_fg_scale_prediction(
        #     light_view=True,
        #     only_padded_image=True,
        #     disable_remove_view=True,
        #     sub_dir=sub_dir
        # )  #

        self.visualize_split_prediction(
            light_view=True,
            only_padded_image=True,
            disable_remove_view=True,
            sub_dir=sub_dir
        )  #

    @staticmethod
    def draw_token_edge(img, input_img_size, feat_map_size):
        # ==========================
        # draw the token boundary
        H, W = input_img_size
        h, w = feat_map_size

        edge_color = tuple([1.0, 0.0, 1.0])  # [1.0, 1.0, 1.0]
        # stride_h, stride_w = H / h, W / w

        stride_h, stride_w = np.ceil(H / h), np.ceil(W / w)
        for ii in range(h):
            y = int(stride_h * ii)
            img = cv2.line(img, (0, y), (W, y), edge_color, 2)

        for ii in range(w):
            x = int(stride_w * ii)
            img = cv2.line(img, (x, 0), (x, H), edge_color, 2)
        # ==========================
        return img

    def draw_gt_box(self, ori_img, boxes, img_size, input_img_size, feat_map_size):
        """
            ori_img = ori_img, boxes = img_target['boxes'],
            img_size = img_target['size'], input_img_size = (H, W),
            feat_map_size = (h, w))
        Args:
            ori_img:
            boxes:
            img_size:
            input_img_size:
            feat_map_size:

        Returns:

        """
        box_img = ori_img.copy()
        num_box = len(boxes)
        box_unnormalized = unnormalize_box(box_normalized=boxes,
                                           input_img_size=img_size)

        if num_box > 0:  # tensor([32.0000, 31.6452], device='cuda:0')
            for box in box_unnormalized:
                x1, y1, x2, y2 = box
                box_img = cv2.rectangle(box_img, (int(x1), int(y1)), (int(x2), int(y2)),
                                        smrc.utils.YELLOW, 3)

        box_img = self.draw_token_edge(img=box_img, input_img_size=input_img_size,
                                       feat_map_size=feat_map_size)
        return box_img

    @staticmethod
    def make_img_size_divisible(img_tensor, feat_map_size):
        """
        conv or pool operation has ceil_mode.

        Args:
            img_tensor: B, C, H, W = batch_img_tensor.shape
            # input_img_size:
            feat_map_size: h, w

        Returns:

        """
        B, C, H, W = img_tensor.shape
        # H, W = input_img_size
        h, w = feat_map_size
        if H / h > H // h or W / w > W // w:
            stride_h, stride_w = int(np.ceil(H / h)), int(np.ceil(W / w))
            H_new, W_new = h * stride_h, w * stride_w
            imgs_new = torch.zeros(B, C, H_new, W_new).to(img_tensor.device)
            imgs_new[..., :H, :W] = img_tensor
            return imgs_new, (H_new, W_new)
        else:
            return img_tensor, (H, W)

    @staticmethod
    def paded_imgs(img_tensor, new_img_size):
        """
        only for gt_raw
        Args:
            img_tensor:
            new_img_size:

        Returns:

        """
        # img_tensor, B, H, W
        B, H, W = img_tensor.shape
        H_new, W_new = new_img_size
        imgs_new = torch.zeros(B, H_new, W_new).to(img_tensor.device)
        imgs_new[..., :H, :W] = img_tensor
        return imgs_new

    @staticmethod
    def resize_token_labels_to_input_img(token_labels, feat_map_size, input_img_size):
        """
        Args:
            token_labels:
            feat_map_size:
            input_img_size:

        Returns:  binary labels

        """
        h, w = feat_map_size
        H, W = input_img_size
        labels = token_labels.permute(1, 0).view(-1, h, w)[:, None, :, :].repeat(
            1, 3, 1, 1)  # N,B -> B, N -> B, h, w -> B, 1, h, w -> B, 3, h, w
        # labels = F.interpolate(labels.float(), [H, W], mode='bilinear').bool().float()  #
        labels = F.interpolate(labels.float(), [H, W], mode='nearest').bool().float()  #
        return labels

    def resize_token_one_hot_label_to_input_img(self, token_labels, feat_map_size, input_img_size):
        """
            pred_labels = pred_labels.view(-1, h, w)[:, None, :, :].repeat(
                1, 3, 1, 1)  #
            pred_labels = F.interpolate(pred_labels.float(), [H, W], mode='nearest')

        Args:
            token_labels:
            feat_map_size:
            input_img_size:

        Returns:

        """
        assert len(token_labels.shape) == 3  # B, N, Num_class
        num_labels = token_labels.shape[-1]
        label_list = []
        for k in range(num_labels):
            # torch.Size([2, 3, 768, 1152])
            label = self.resize_token_labels_to_input_img(token_labels[:, :, k], feat_map_size, input_img_size)
            label_list.append(label)
        final_labels = torch.stack(label_list, dim=-1)

        return final_labels

    @staticmethod
    def resized_token_label_to_img_array(single_img_labels, color=smrc.utils.WHITE):
        """
        Permute the dimension to image array format, and fill in color for each 1 location.
        Args:
            single_img_labels:
            color:

        Returns:

        """
        img = single_img_labels.permute(1, 2, 0).detach().cpu().float().numpy()
        final_img = np.float32(img * np.array(color))
        return final_img

    @staticmethod
    def resized_token_one_hot_label_to_color_img(pred_mask):
        """
                pred_img = pred_mask[k].permute(1, 2, 0).cpu().float().numpy()  # * 255
        pred_img = pred_img * np.array(smrc.utils.color.RED).reshape(3, 1, 1) + \
                   (1 - pred_img) * np.array(smrc.utils.color.YELLOW).reshape(3, 1, 1)
        Args:
            pred_mask:

        Returns:

        """
        num_label = pred_mask.shape[-1]
        # colors = smrc.utils.color.color_map
        colors = [
            np.array(smrc.utils.color.WHITE),
            np.array(smrc.utils.color.GREEN),  # missed
            np.array(smrc.utils.color.Pink),  # missed bg
        ]
        # 3, H, W, Num_label -> H, W, 3, Num_label (1105, 736, 3, 2)
        pred_img = pred_mask.detach().cpu().permute(1, 2, 0, 3).float().numpy()  # * 255
        mask = np.zeros_like(pred_img[:, :, :, 0])  # H, W, 3,

        for k in range(num_label):
            class_mask = pred_img[:, :, :, k] == 1  # (1105, 736, 3)

            mask += class_mask * colors[k].reshape(1, 1, 3)
        return mask

    def _resize_list_of_tokens(self, list_of_tokens, feat_map_size, input_img_size):
        return [self.resize_token_labels_to_input_img(x, feat_map_size=feat_map_size, input_img_size=input_img_size)
                for x in list_of_tokens]

    def _resize_list_of_one_hot_tokens(self, list_of_tokens, feat_map_size, input_img_size):
        return [self.resize_token_one_hot_label_to_input_img(x, feat_map_size=feat_map_size,
                                                             input_img_size=input_img_size)
                for x in list_of_tokens]

    @staticmethod
    def _get_pred_fg_tokens_classified(tokens_to_discard, fg_gt):
        bg_tokens_correct, fg_tokens_missed, bg_tokens_missed = classify_predicted_remove_tokens(
            tokens_to_discard=tokens_to_discard, fg_gt=fg_gt)
        # torch.Size([864, 2, 3]) N, B, 3
        fg_pred_tokens_classified = torch.stack((bg_tokens_correct, fg_tokens_missed, bg_tokens_missed), dim=-1)
        return fg_pred_tokens_classified

    @staticmethod
    def _get_pred_split_tokens_classified(tokens_to_split, split_gt):
        split_tokens_correct, false_split_tokens, split_tokens_missed = classify_predicted_split_tokens(
            tokens_to_split=tokens_to_split, split_gt=split_gt)
        split_pred_tokens_classified = torch.stack(
            (split_tokens_correct, false_split_tokens, split_tokens_missed), dim=-1)
        return split_pred_tokens_classified

    def _draw_token_label_on_img_and_save(self, img, token_label_single_img, img_path, one_hot_label=False):
        """
        Draw the token label on the original image by overlapping, and save the plot to file.
        Args:
            img:
            token_label_single_img:
            img_path:

        Returns:

        """
        if not one_hot_label:
            feat_img = self.resized_token_label_to_img_array(token_label_single_img)
        else:
            feat_img = self.resized_token_one_hot_label_to_color_img(token_label_single_img)

        final_img = cv2.addWeighted(img.copy(), 0.5, feat_img, 0.5, 2.2)
        cv2.imwrite(img_path, final_img)

    def _get_file_path(self, file_path):
        return os.path.join(self.out_dir, file_path)

    def _draw_box_img(self, ori_img, img_target, input_img_size, feat_map_size, img_path=None):
        box_img = self.draw_gt_box(
            ori_img=ori_img, boxes=img_target['boxes'],
            img_size=img_target['size'],
            input_img_size=input_img_size,
            feat_map_size=feat_map_size
        )
        if img_path is None:
            return box_img
        else:
            if not os.path.isfile(img_path):
                cv2.imwrite(img_path, box_img)
            return box_img

    @staticmethod
    def _draw_significace_score(gt_sig_value_matrix, pred_sig_value_matrix, plot_name):
        # define a list of markevery cases to plot
        thds = np.linspace(0, 1.0, 9)

        # data points
        fig, axs = plt.subplots(3, 3, figsize=(10, 6))  # , constrained_layout=True
        for k, (ax, thd) in enumerate(zip(axs.flat, thds[:9])):
            if k == 0:
                matrix = gt_sig_value_matrix.copy()
                ax.set_title(f'gt')
            else:
                matrix = np.where(pred_sig_value_matrix > thds[k - 1], pred_sig_value_matrix, 0)
                ax.set_title(f'thd={thds[k - 1]}')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(matrix, vmin=0, vmax=1.0)  #
            fig.colorbar(im, cax=cax, orientation='vertical')
            # fig.colorbar(ax=ax)
            # ax.plot(x, y, 'o', ls='-', ms=4, markevery=markevery)
        plt.tight_layout()
        plt.savefig(plot_name)
        plt.close('all')

    def save_intermediate_result(self):
        # result_file_path = #
        result_file_path = ''
        for target in self.targets:
            img_id = target['image_id'].detach().cpu().item()
            result_file_path += f'_{img_id}'

        result_path = os.path.join(self.out_dir, f'{result_file_path}.pkl')
        smrc.utils.generate_pkl_file(pkl_file_name=result_path, data=[
            self.targets,
            self.sgdt_target_raw,
            self.sgdt_output,
            self.sgdt_targets,
        ])

    def visualize_fg_scale_prediction(
            self, disable_remove_view=False, disable_split_view=False, light_view=False,
            only_padded_image=False, sub_dir=None

    ):
        """
            # ---------------- for visualization purpose
            fg_mask=adapted_token_dict['fg_mask'],
            fg_obj_score=adapted_token_dict['fg_obj_score'],
            tokens_to_discard_original=adapted_token_dict['tokens_to_discard_original'],
            small_scale_score=adapted_token_dict['small_scale_score'],
            scale_mask=adapted_token_dict['scale_mask'],
            tokens_to_split_original=adapted_token_dict['tokens_to_split_original'],
        Returns:

        """
        if sub_dir is None:
            out_dir = self.out_dir
        else:
            out_dir = os.path.join(self.out_dir, sub_dir)
        smrc.utils.generate_dir_if_not_exist(out_dir)

        significance_score = self.sgdt_output['significance_score']  # N, B
        tokens_to_discard_original = self.sgdt_output['tokens_to_discard_original'].long()
        tokens_to_split_original = self.sgdt_output['tokens_to_split_original'].long()
        # bg_mask, scale_mask = F.one_hot(tokens_to_discard_original, num_classes=2).float(), \
        #                       F.one_hot(tokens_to_split_original, num_classes=2).float()

        tokens_small_obj = self.sgdt_output['tokens_small_obj']
        tokens_to_discard = self.sgdt_output['tokens_to_discard']

        img = self.targets[-1]['input_imgs']
        h, w = list(self.sgdt_output['feat_map_size'].detach().cpu().numpy())
        valid_tokens = self.sgdt_output['valid_tokens'].bool()  # N, B
        # new_img_size (H_new, W_new)
        img, new_img_size = self.make_img_size_divisible(img, feat_map_size=(h, w))

        B, C, H, W = img.shape  # (Tensor[B, 3, H, W])
        # -------------------- raw gt
        fg_gt_raw, scale_gt_raw = self.sgdt_target_raw['fg_gt'], self.sgdt_target_raw['scale_gt']
        # update the gt_raw if the original imgs are updated
        fg_gt_raw = self.paded_imgs(fg_gt_raw, new_img_size=new_img_size)
        scale_gt_raw = self.paded_imgs(scale_gt_raw, new_img_size=new_img_size)
        fg_gt, scale_gt = self.sgdt_targets['fg_gt'], self.sgdt_targets['scale_gt']

        # -----------------------------------------------
        fg_pred_tokens_classified = self._get_pred_fg_tokens_classified(
            tokens_to_discard=tokens_to_discard, fg_gt=fg_gt)
        fg_original_pred_tokens_classified = self._get_pred_fg_tokens_classified(
            tokens_to_discard=tokens_to_discard_original, fg_gt=fg_gt)
        split_pred_tokens_classified = self._get_pred_split_tokens_classified(
            tokens_to_split=tokens_small_obj, split_gt=scale_gt)
        split_original_pred_tokens_classified = self._get_pred_split_tokens_classified(
            tokens_to_split=tokens_to_split_original, split_gt=scale_gt)
        # ------------------------------------------------- plot section

        for i, (gt_name, gt, gt_on_feat_map,
                # pred_labels, #
                tokens_pred_classified, tokens_original_pred_classified,
                pred_sampled_token_label, pred_final_token_label) in enumerate(zip(
            ['discard', 'split'],
            [fg_gt_raw, scale_gt_raw],
            [fg_gt, scale_gt],
            [fg_pred_tokens_classified, split_pred_tokens_classified],
            [fg_original_pred_tokens_classified, split_original_pred_tokens_classified],
            [tokens_to_discard_original, tokens_to_split_original],
            [tokens_to_discard, tokens_small_obj],
        )):
            if i == 0 and disable_remove_view:
                continue
            elif i == 1 and disable_split_view:
                continue

            gt_raw = gt[:, None, :, :].repeat(1, 3, 1, 1)  # .bool().float()  # B, H, W -> B, 3, H, W,

            gt_feat_map, pred_sampled_token_label, pred_final_token_label = \
                self._resize_list_of_tokens(
                    list_of_tokens=[gt_on_feat_map, pred_sampled_token_label,
                                    pred_final_token_label],
                    input_img_size=(H, W), feat_map_size=(h, w)
                )

            pred_significance_score = significance_score.permute(1, 0).view(-1, h, w)[:, None, :, :].repeat(
                1, 3, 1, 1)  # N,B -> B, 3, H, W,
            pred_significance_score = F.interpolate(pred_significance_score.float(), [H, W],
                                                    mode='nearest')  # torch.Size([2, 3, 1105, 736])

            tokens_pred_classified, tokens_original_pred_classified = \
                self._resize_list_of_one_hot_tokens(
                    list_of_tokens=[tokens_pred_classified, tokens_original_pred_classified],
                    input_img_size=(H, W), feat_map_size=(h, w)
                )

            # if fg_missed is not None:
            #     fg_missed = self.resize_tokens_to_input_img(
            #         fg_missed, feat_map_size=(h, w), input_img_size=(H, W))

            for k in range(B):
                # skip unpadded images.
                if only_padded_image and valid_tokens[:, k].all():
                    continue

                img_id = self.targets[k]['image_id'].detach().cpu().item()
                ori_img = img[k].permute(1, 2, 0).detach().cpu().numpy() * 255
                # img_path = os.path.join(out_dir, f'{img_id}_original_img.jpg')
                # if not os.path.isfile(img_path):
                #     cv2.imwrite(img_path, ori_img)

                img_path = os.path.join(out_dir, f'{img_id}_box_img.jpg')
                box_img = self._draw_box_img(ori_img=ori_img, img_target=self.targets[k], input_img_size=(H, W),
                                             feat_map_size=(h, w), img_path=img_path)

                # ---------------- plot gt token labels on the input image
                self._draw_token_label_on_img_and_save(
                    img=box_img, token_label_single_img=gt_feat_map[k],
                    img_path=os.path.join(out_dir, f'{img_id}_gt_{gt_name}_overlap.jpg')
                )

                self._draw_token_label_on_img_and_save(
                    img=box_img, token_label_single_img=gt_raw[k],
                    img_path=os.path.join(out_dir, f'{img_id}_gt_{gt_name}_raw_overlap.jpg')
                )

                self._draw_token_label_on_img_and_save(
                    img=box_img, token_label_single_img=tokens_pred_classified[k],
                    img_path=os.path.join(out_dir, f'{img_id}_pred_{gt_name}_mask_overlap.jpg'),
                    one_hot_label=True)
                self._draw_token_label_on_img_and_save(
                    img=box_img, token_label_single_img=tokens_original_pred_classified[k],
                    img_path=os.path.join(out_dir, f'{img_id}_pred_{gt_name}_mask_raw_overlap.jpg'),
                    one_hot_label=True)

                self._draw_token_label_on_img_and_save(
                    img=box_img, token_label_single_img=pred_sampled_token_label[k],
                    img_path=os.path.join(out_dir, f'{img_id}_pred_{gt_name}_token_label_raw_sampled_overlap.jpg')
                )
                self._draw_token_label_on_img_and_save(
                    img=box_img, token_label_single_img=pred_final_token_label[k],
                    img_path=os.path.join(out_dir, f'{img_id}_pred_{gt_name}_token_label_final_overlap.jpg')
                )
                # --------------------------- significance_score
                img_path_pred = os.path.join(out_dir, f'{img_id}_significance_score_pred_{gt_name}.jpg')
                smrc.utils.plot_matrix(
                    matrix=pred_significance_score[k][0].detach().cpu().numpy(),  # (3, 704, 1072)
                    vmin=0, vmax=1.0,
                    plot_name=img_path_pred)
                img_path_gt = os.path.join(out_dir, f'{img_id}_significance_score_gt_{gt_name}.jpg')
                smrc.utils.plot_matrix(
                    matrix=gt[k].detach().cpu().numpy(),  # (3, 704, 1072)
                    vmin=0, vmax=1.0,
                    plot_name=img_path_gt,
                )
                gt_final_img = cv2.imread(img_path_gt)
                pred_final_img = cv2.imread(img_path_pred)
                compare_img = smrc.utils.merging_two_images_side_by_side(
                    gt_final_img, pred_final_img
                )
                compare_img = smrc.utils.merging_two_images_side_by_side(
                    box_img, compare_img
                )
                img_path = os.path.join(out_dir, f'{img_id}_gt_vs_pred_{gt_name}.jpg')
                cv2.imwrite(img_path, compare_img)

                # if not light_view:
                self._draw_significace_score(
                    gt_sig_value_matrix=gt[k].detach().cpu().numpy(),
                    pred_sig_value_matrix=pred_significance_score[k][0].detach().cpu().numpy(),
                    plot_name=os.path.join(out_dir, f'{img_id}_pred_{gt_name}_significance_score_vs_gt.jpg')
                )

                # self._draw_token_label_on_img_and_save(
                #     img=box_img, token_label_single_img=pred_significance_score[k],
                #     img_path=os.path.join(out_dir, f'{img_id}_pred_{gt_name}_token_label_raw_sampled_overlap.jpg')
                # )
                # pred_img = self.token_score_feat_map_to_img(pred_significance_score[k])
                # # img_path = os.path.join(out_dir, f'{img_id}_gt_{gt_name}.jpg')
                # # cv2.imwrite(img_path, feat_img)
                # pred_final_img = cv2.addWeighted(box_img.copy(), 0.5, pred_img, 0.5, 2.2)
                # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_overlap.jpg')
                # cv2.imwrite(img_path, pred_final_img)
                # compare_img = smrc.utils.merging_two_images_side_by_side(
                #     gt_final_img, pred_final_img
                # )
                # img_path = os.path.join(out_dir, f'{img_id}_gt_vs_pred_{gt_name}.jpg')
                # cv2.imwrite(img_path, compare_img)

                # pred_img = pred_mask[k].permute(1, 2, 0).cpu().float().numpy() # * 255
                # pred_img = pred_img * np.array(smrc.utils.color.RED).reshape(3, 1, 1) + \
                #            (1 - pred_img) * np.array(smrc.utils.color.YELLOW).reshape(3, 1, 1)

                # # plot missed fg tokens
                # if fg_missed is not None:
                #     pred_img = self.resized_token_label_to_img_array(fg_missed[k])
                #     final_img = cv2.addWeighted(box_img.copy(), 0.5, pred_img, 0.5, 2.2)
                #     img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_missed.jpg')
                #     cv2.imwrite(img_path, final_img)

    def visualize_split_prediction(
            self, disable_remove_view=False, disable_split_view=False, light_view=False,
            only_padded_image=False, sub_dir=None

    ):
        """
            # ---------------- for visualization purpose
            fg_mask=adapted_token_dict['fg_mask'],
            fg_obj_score=adapted_token_dict['fg_obj_score'],
            tokens_to_discard_original=adapted_token_dict['tokens_to_discard_original'],
            small_scale_score=adapted_token_dict['small_scale_score'],
            scale_mask=adapted_token_dict['scale_mask'],
            tokens_to_split_original=adapted_token_dict['tokens_to_split_original'],
        Returns:

        """
        if sub_dir is None:
            out_dir = self.out_dir
        else:
            out_dir = os.path.join(self.out_dir, sub_dir)
        smrc.utils.generate_dir_if_not_exist(out_dir)

        significance_score = self.sgdt_output['significance_score']  # N, B
        tokens_to_discard_original = self.sgdt_output['tokens_to_discard_original'].long()
        tokens_to_split_original = self.sgdt_output['tokens_to_split_original'].long()
        # bg_mask, scale_mask = F.one_hot(tokens_to_discard_original, num_classes=2).float(), \
        #                       F.one_hot(tokens_to_split_original, num_classes=2).float()

        tokens_small_obj = self.sgdt_output['tokens_small_obj']
        tokens_to_discard = self.sgdt_output['tokens_to_discard']

        img = self.targets[-1]['input_imgs']
        h, w = list(self.sgdt_output['feat_map_size'].detach().cpu().numpy())
        valid_tokens = self.sgdt_output['valid_tokens'].bool()  # N, B
        # new_img_size (H_new, W_new)
        img, new_img_size = self.make_img_size_divisible(img, feat_map_size=(h, w))

        B, C, H, W = img.shape  # (Tensor[B, 3, H, W])
        # -------------------- raw gt
        fg_gt_raw, scale_gt_raw = self.sgdt_target_raw['fg_gt'], self.sgdt_target_raw['scale_gt']
        # update the gt_raw if the original imgs are updated
        fg_gt_raw = self.paded_imgs(fg_gt_raw, new_img_size=new_img_size)
        scale_gt_raw = self.paded_imgs(scale_gt_raw, new_img_size=new_img_size)
        fg_gt, scale_gt = self.sgdt_targets['fg_gt'], self.sgdt_targets['scale_gt']

        # -----------------------------------------------
        fg_pred_tokens_classified = self._get_pred_fg_tokens_classified(
            tokens_to_discard=tokens_to_discard, fg_gt=fg_gt)
        fg_original_pred_tokens_classified = self._get_pred_fg_tokens_classified(
            tokens_to_discard=tokens_to_discard_original, fg_gt=fg_gt)
        split_pred_tokens_classified = self._get_pred_split_tokens_classified(
            tokens_to_split=tokens_small_obj, split_gt=scale_gt)
        split_original_pred_tokens_classified = self._get_pred_split_tokens_classified(
            tokens_to_split=tokens_to_split_original, split_gt=scale_gt)
        # ------------------------------------------------- plot section

        for i, (gt_name, gt, gt_on_feat_map,
                # pred_labels, #
                tokens_pred_classified, tokens_original_pred_classified,
                pred_sampled_token_label, pred_final_token_label) in enumerate(zip(
            ['discard', 'split'],
            [fg_gt_raw, scale_gt_raw],
            [fg_gt, scale_gt],
            [fg_pred_tokens_classified, split_pred_tokens_classified],
            [fg_original_pred_tokens_classified, split_original_pred_tokens_classified],
            [tokens_to_discard_original, tokens_to_split_original],
            [tokens_to_discard, tokens_small_obj],
        )):
            if i == 0 and disable_remove_view:
                continue
            elif i == 1 and disable_split_view:
                continue

            gt_raw = gt[:, None, :, :].repeat(1, 3, 1, 1)  # .bool().float()  # B, H, W -> B, 3, H, W,

            gt_feat_map, pred_sampled_token_label, pred_final_token_label = \
                self._resize_list_of_tokens(
                    list_of_tokens=[gt_on_feat_map, pred_sampled_token_label,
                                    pred_final_token_label],
                    input_img_size=(H, W), feat_map_size=(h, w)
                )

            pred_significance_score = significance_score.permute(1, 0).view(-1, h, w)[:, None, :, :].repeat(
                1, 3, 1, 1)  # N,B -> B, 3, H, W,
            pred_significance_score = F.interpolate(pred_significance_score.float(), [H, W],
                                                    mode='nearest')  # torch.Size([2, 3, 1105, 736])

            tokens_pred_classified, tokens_original_pred_classified = \
                self._resize_list_of_one_hot_tokens(
                    list_of_tokens=[tokens_pred_classified, tokens_original_pred_classified],
                    input_img_size=(H, W), feat_map_size=(h, w)
                )

            # if fg_missed is not None:
            #     fg_missed = self.resize_tokens_to_input_img(
            #         fg_missed, feat_map_size=(h, w), input_img_size=(H, W))

            for k in range(B):
                # skip unpadded images.
                if only_padded_image and valid_tokens[:, k].all():
                    continue

                img_id = self.targets[k]['image_id'].detach().cpu().item()
                ori_img = img[k].permute(1, 2, 0).detach().cpu().numpy() * 255
                # img_path = os.path.join(out_dir, f'{img_id}_original_img.jpg')
                # if not os.path.isfile(img_path):
                #     cv2.imwrite(img_path, ori_img)

                # img_path = os.path.join(out_dir, f'{img_id}_box_img.jpg')
                box_img = self._draw_box_img(ori_img=ori_img, img_target=self.targets[k], input_img_size=(H, W),
                                             feat_map_size=(h, w)
                                             # , img_path=img_path
                                             )

                # ---------------- plot gt token labels on the input image
                # self._draw_token_label_on_img_and_save(
                #     img=box_img, token_label_single_img=gt_feat_map[k],
                #     img_path=os.path.join(out_dir, f'{img_id}_gt_{gt_name}_overlap.jpg')
                # )

                # self._draw_token_label_on_img_and_save(
                #     img=box_img, token_label_single_img=gt_raw[k],
                #     img_path=os.path.join(out_dir, f'{img_id}_gt_{gt_name}_raw_overlap.jpg')
                # )

                # self._draw_token_label_on_img_and_save(
                #     img=box_img, token_label_single_img=tokens_pred_classified[k],
                #     img_path=os.path.join(out_dir, f'{img_id}_pred_{gt_name}_mask_overlap.jpg'),
                #     one_hot_label=True)
                # self._draw_token_label_on_img_and_save(
                #     img=box_img, token_label_single_img=tokens_original_pred_classified[k],
                #     img_path=os.path.join(out_dir, f'{img_id}_pred_{gt_name}_mask_raw_overlap.jpg'),
                #     one_hot_label=True)

                # self._draw_token_label_on_img_and_save(
                #     img=box_img, token_label_single_img=pred_sampled_token_label[k],
                #     img_path=os.path.join(out_dir, f'{img_id}_pred_{gt_name}_token_label_raw_sampled_overlap.jpg')
                # )
                self._draw_token_label_on_img_and_save(
                    img=box_img, token_label_single_img=pred_final_token_label[k],
                    img_path=os.path.join(out_dir, f'{img_id}_pred_{gt_name}_token_label_final_overlap.jpg')
                )
                # --------------------------- significance_score
                # img_path_pred = os.path.join(out_dir, f'{img_id}_significance_score_pred_{gt_name}.jpg')
                # smrc.utils.plot_matrix(
                #     matrix=pred_significance_score[k][0].detach().cpu().numpy(),  # (3, 704, 1072)
                #     vmin=0, vmax=1.0,
                #     plot_name=img_path_pred)
                # img_path_gt = os.path.join(out_dir, f'{img_id}_significance_score_gt_{gt_name}.jpg')
                # smrc.utils.plot_matrix(
                #     matrix=gt[k].detach().cpu().numpy(),  # (3, 704, 1072)
                #     vmin=0, vmax=1.0,
                #     plot_name=img_path_gt,
                # )
                # gt_final_img = cv2.imread(img_path_gt)
                # pred_final_img = cv2.imread(img_path_pred)
                # compare_img = smrc.utils.merging_two_images_side_by_side(
                #     gt_final_img, pred_final_img
                # )
                # compare_img = smrc.utils.merging_two_images_side_by_side(
                #     box_img, compare_img
                # )
                # img_path = os.path.join(out_dir, f'{img_id}_gt_vs_pred_{gt_name}.jpg')
                # cv2.imwrite(img_path, compare_img)
                #
                # # if not light_view:
                # self._draw_significace_score(
                #     gt_sig_value_matrix=gt[k].detach().cpu().numpy(),
                #     pred_sig_value_matrix=pred_significance_score[k][0].detach().cpu().numpy(),
                #     plot_name=os.path.join(out_dir, f'{img_id}_pred_{gt_name}_significance_score_vs_gt.jpg')
                # )

                # self._draw_token_label_on_img_and_save(
                #     img=box_img, token_label_single_img=pred_significance_score[k],
                #     img_path=os.path.join(out_dir, f'{img_id}_pred_{gt_name}_token_label_raw_sampled_overlap.jpg')
                # )
                # pred_img = self.token_score_feat_map_to_img(pred_significance_score[k])
                # # img_path = os.path.join(out_dir, f'{img_id}_gt_{gt_name}.jpg')
                # # cv2.imwrite(img_path, feat_img)
                # pred_final_img = cv2.addWeighted(box_img.copy(), 0.5, pred_img, 0.5, 2.2)
                # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_overlap.jpg')
                # cv2.imwrite(img_path, pred_final_img)
                # compare_img = smrc.utils.merging_two_images_side_by_side(
                #     gt_final_img, pred_final_img
                # )
                # img_path = os.path.join(out_dir, f'{img_id}_gt_vs_pred_{gt_name}.jpg')
                # cv2.imwrite(img_path, compare_img)

                # pred_img = pred_mask[k].permute(1, 2, 0).cpu().float().numpy() # * 255
                # pred_img = pred_img * np.array(smrc.utils.color.RED).reshape(3, 1, 1) + \
                #            (1 - pred_img) * np.array(smrc.utils.color.YELLOW).reshape(3, 1, 1)

                # # plot missed fg tokens
                # if fg_missed is not None:
                #     pred_img = self.resized_token_label_to_img_array(fg_missed[k])
                #     final_img = cv2.addWeighted(box_img.copy(), 0.5, pred_img, 0.5, 2.2)
                #     img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_missed.jpg')
                #     cv2.imwrite(img_path, final_img)

    def visualize_l1_loss_prediction(self):
        """
            # ---------------- for visualization purpose
            fg_mask=adapted_token_dict['fg_mask'],
            fg_obj_score=adapted_token_dict['fg_obj_score'],
            tokens_to_discard_original=adapted_token_dict['tokens_to_discard_original'],
            small_scale_score=adapted_token_dict['small_scale_score'],
            scale_mask=adapted_token_dict['scale_mask'],
            tokens_to_split_original=adapted_token_dict['tokens_to_split_original'],
        Returns:

        """
        out_dir = self.out_dir

        significance_score = self.sgdt_output['significance_score']  # N, B
        tokens_to_discard_original = self.sgdt_output['tokens_to_discard_original'].long()
        tokens_to_split_original = self.sgdt_output['tokens_to_split_original'].long()
        bg_mask, scale_mask = F.one_hot(tokens_to_discard_original, num_classes=2).float(), \
                              F.one_hot(tokens_to_split_original, num_classes=2).float()

        tokens_small_obj = self.sgdt_output['tokens_small_obj']
        tokens_to_discard = self.sgdt_output['tokens_to_discard']

        # -------------------- raw gt
        img = self.targets[-1]['input_imgs']
        B, C, H, W = img.shape  # (Tensor[B, 3, H, W])
        fg_gt_raw, scale_gt_raw = self.sgdt_target_raw['fg_gt'], self.sgdt_target_raw['scale_gt']

        # -------------------- gt of feature map size
        h, w = list(self.sgdt_output['feat_map_size'].detach().cpu().numpy())
        # sgdt_targets = resize_sgdt_target(self.sgdt_target_raw,
        #                                   feat_map_size=self.sgdt_output['feat_map_size'],
        #                                   interpolate_mode='bilinear'
        #                                   )
        fg_gt, scale_gt = self.sgdt_targets['fg_gt'], self.sgdt_targets['scale_gt']
        N, _ = fg_gt.shape

        # ------------------------------------------------- plot section
        for i, (gt_name, gt, gt_on_feat_map,
                # pred_labels, # pred_mask,
                pred_sampled_token_label, pred_final_token_label) in enumerate(zip(
            ['discard', 'split'],
            [fg_gt_raw, scale_gt_raw],
            [fg_gt, scale_gt],
            # [bg_mask, scale_mask],
            [tokens_to_discard_original, tokens_to_split_original],
            [tokens_to_discard, tokens_small_obj],
        )):
            gt_raw = gt[:, None, :, :].repeat(1, 3, 1, 1)  # .bool().float()  # B, H, W -> B, 3, H, W,

            # plot gt_feat_map on the input image
            gt_feat_map = gt_on_feat_map.permute(1, 0).view(-1, h, w)[:, None, :, :].repeat(
                1, 3, 1, 1)  # N,B -> B, 3, H, W,
            # We should use the 'nearest' mode to increase the resoltion, otherwise, by 'bilinear', we will cause more
            # non-zero locations.
            gt_feat_map = F.interpolate(gt_feat_map.float(), [H, W],
                                        mode='nearest')  # torch.Size([2, 3, 1105, 736])
            # gt_feat_map = F.interpolate(gt_feat_map.float(), [H, W], mode='bilinear').bool().float()  # torch.Size([2, 3, 1105, 736])

            pred_significance_score = significance_score.permute(1, 0).view(-1, h, w)[:, None, :, :].repeat(
                1, 3, 1, 1)  # N,B -> B, 3, H, W,
            pred_significance_score = F.interpolate(pred_significance_score.float(), [H, W],
                                                    mode='nearest')  # torch.Size([2, 3, 1105, 736])

            # pred_labels = self.resize_token_label_to_input_img(pred_labels, feat_map_size=(h, w), input_img_size=(H, W))
            # # -----------------------------
            # pred_mask = self.resize_token_multi_label_to_input_img(
            #     pred_mask, feat_map_size=(h, w), input_img_size=(H, W))

            pred_sampled_token_label = self.resize_token_labels_to_input_img(
                pred_sampled_token_label, feat_map_size=(h, w), input_img_size=(H, W))
            pred_final_token_label = self.resize_token_labels_to_input_img(
                pred_final_token_label, feat_map_size=(h, w), input_img_size=(H, W))

            for k in range(B):
                img_id = self.targets[k]['image_id'].detach().cpu().item()
                ori_img = img[k].permute(1, 2, 0).detach().cpu().numpy() * 255

                img_target = self.targets[k]
                box_img = self.draw_gt_box(
                    ori_img=ori_img, boxes=img_target['boxes'],
                    img_size=img_target['size'], input_img_size=(H, W),
                    feat_map_size=(h, w))
                # TODO: check input_img_size=(H, W)

                img_path = os.path.join(out_dir, f'{img_id}_box_img.jpg')
                if not os.path.isfile(img_path):
                    cv2.imwrite(img_path, box_img)

                ori_img = box_img.copy()
                # img_path = os.path.join(out_dir, f'{img_id}_original_img.jpg')
                # if not os.path.isfile(img_path):
                #     cv2.imwrite(img_path, ori_img)

                # feat_img = feat_map[k].permute(1, 2, 0).cpu().numpy() * 255
                feat_img = gt_raw[k].permute(1, 2, 0).cpu().float().numpy() * 255
                # img_path = os.path.join(out_dir, f'{img_id}_feat_img_raw_{gt_name}.jpg')
                # cv2.imwrite(img_path, feat_img)
                final_img = cv2.addWeighted(ori_img, 0.5, feat_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_raw_gt_{gt_name}_overlap.jpg')
                cv2.imwrite(img_path, final_img)

                # ---------------- plot gt token labels on the input image
                feat_img = self.resized_token_label_to_img_array(gt_feat_map[k])
                # feat_img = self.token_score_feat_map_to_img(gt_feat_map[k])
                # img_path = os.path.join(out_dir, f'{img_id}_gt_{gt_name}.jpg')
                # cv2.imwrite(img_path, feat_img)
                gt_final_img = cv2.addWeighted(box_img.copy(), 0.5, feat_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_gt_{gt_name}_overlap.jpg')
                cv2.imwrite(img_path, gt_final_img)

                # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_significance_score.jpg')
                # smrc.utils.plot_matrix(
                #     matrix=pred_significance_score[k][0].detach().cpu().numpy(),  # (3, 704, 1072)
                #     plot_name=img_path)
                #
                # # pred_img = self.token_score_feat_map_to_img(pred_significance_score[k])
                # # # img_path = os.path.join(out_dir, f'{img_id}_gt_{gt_name}.jpg')
                # # # cv2.imwrite(img_path, feat_img)
                # # pred_final_img = cv2.addWeighted(box_img.copy(), 0.5, pred_img, 0.5, 2.2)
                # # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_overlap.jpg')
                # # cv2.imwrite(img_path, pred_final_img)
                # # compare_img = smrc.utils.merging_two_images_side_by_side(
                # #     gt_final_img, pred_final_img
                # # )
                # # img_path = os.path.join(out_dir, f'{img_id}_gt_vs_pred_{gt_name}.jpg')
                # # cv2.imwrite(img_path, compare_img)
                #
                # # # pred_img = pred_labels[k].permute(1, 2, 0).detach().cpu().float().numpy() * 255
                # # pred_img = self.token_label_feat_map_to_img(pred_labels[k])
                # # # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}.jpg')
                # # # cv2.imwrite(img_path, pred_img)
                # # final_img = cv2.addWeighted(ori_img, 0.5, pred_img, 0.5, 2.2)
                # # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_overlap.jpg')
                # # cv2.imwrite(img_path, final_img)
                #
                #
                #
                # # # pred_img = pred_mask[k].permute(1, 2, 0).cpu().float().numpy() # * 255
                # # # pred_img = pred_img * np.array(smrc.utils.color.RED).reshape(3, 1, 1) + \
                # # #            (1 - pred_img) * np.array(smrc.utils.color.YELLOW).reshape(3, 1, 1)
                # # pred_img = self.token_mask_to_color_img(pred_mask[k])
                # # # fg red, bg yellow.
                # # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_mask.jpg')
                # # cv2.imwrite(img_path, pred_img)
                # # final_img = cv2.addWeighted(ori_img, 0.5, pred_img, 0.5, 2.2)
                # # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_mask_overlap.jpg')
                # # cv2.imwrite(img_path, final_img)
                # # --------------
                # pred_img = self.token_label_feat_map_to_img(pred_sampled_token_label[k])
                # # # self.token_mask_to_color_img(pred_sampled_token_label[k])
                # # # fg red, bg yellow.
                # # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_pred_raw_sampled_token_label.jpg')
                # # cv2.imwrite(img_path, pred_img)
                # final_img = cv2.addWeighted(box_img.copy(), 0.5, pred_img, 0.5, 2.2)
                # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_raw_sampled_token_label_overlap.jpg')
                # cv2.imwrite(img_path, final_img)
                #
                # pred_img = self.token_label_feat_map_to_img(pred_final_token_label[k])
                # # # fg red, bg yellow.
                # # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_pred_final_token_label.jpg')
                # # cv2.imwrite(img_path, pred_img)
                # final_img = cv2.addWeighted(box_img.copy(), 0.5, pred_img, 0.5, 2.2)
                # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_final_token_label_overlap.jpg')
                # cv2.imwrite(img_path, final_img)
                #
                # # draw predicted significant scores.

    def visualize_fg_scale_gt(self):
        out_dir = self.out_dir

        # -------------------- plot on image of input size
        img = self.targets[-1]['input_imgs']
        B, C, H, W = img.shape  # (Tensor[B, 3, H, W])
        fg_gt_raw, scale_gt_raw = self.sgdt_target_raw['fg_gt'], self.sgdt_target_raw['scale_gt']

        # -------------------- plot on image of feature map size
        h, w = list(self.sgdt_output['feat_map_size'].cpu().numpy())
        sgdt_targets = resize_sgdt_target(self.sgdt_target_raw, feat_map_size=self.sgdt_output['feat_map_size'])
        fg_gt, scale_gt = sgdt_targets['fg_gt'], sgdt_targets['scale_gt']
        N, _ = fg_gt.shape
        # -------------------------------------------------

        for id, (gt, gt_name, gt_on_feat_map) in enumerate(zip([fg_gt_raw, scale_gt_raw],
                                                               ['fg_gt', 'scale_gt'], [fg_gt, scale_gt])):
            gt_raw = gt[:, None, :, :].repeat(1, 3, 1, 1)  # B, H, W -> B, 3, H, W,

            gt_feat_map = gt_on_feat_map.permute(1, 0).view(-1, h, w)[:, None, :, :].repeat(1, 3, 1,
                                                                                            1)  # N,B -> B, 3, H, W,
            gt_feat_map = F.interpolate(gt_feat_map.float(), [H, W], mode='bilinear').bool().float()

            for k in range(B):
                img_id = self.targets[k]['image_id'].cpu().item()
                ori_img = img[k].permute(1, 2, 0).cpu().numpy() * 255

                box_img = ori_img.copy()
                img_target = self.targets[k]
                num_box = len(img_target['boxes'])
                box_unnormalized = unnormalize_box(box_normalized=img_target['boxes'],
                                                   input_img_size=img_target['size'])

                if num_box > 0:  # tensor([32.0000, 31.6452], device='cuda:0')
                    for box in box_unnormalized:
                        x1, y1, x2, y2 = box
                        box_img = cv2.rectangle(box_img, (int(x1), int(y1)), (int(x2), int(y2)), smrc.utils.YELLOW, 4)

                box_img = self.draw_token_edge(img=box_img, input_img_size=(H, W), feat_map_size=(h, w))

                img_path = os.path.join(out_dir, f'{img_id}_box_img.jpg')
                if not os.path.isfile(img_path):
                    cv2.imwrite(img_path, box_img)

                ori_img = box_img.copy()
                img_path = os.path.join(out_dir, f'{img_id}_original_img.jpg')
                if not os.path.isfile(img_path):
                    cv2.imwrite(img_path, ori_img)

                # feat_img = feat_map[k].permute(1, 2, 0).cpu().numpy() * 255
                feat_img = gt_raw[k].permute(1, 2, 0).cpu().float().numpy() * 255
                img_path = os.path.join(out_dir, f'{img_id}_feat_img_raw_{gt_name}.jpg')
                cv2.imwrite(img_path, feat_img)
                final_img = cv2.addWeighted(ori_img, 0.5, feat_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_final_img_raw_{gt_name}_overlap.jpg')
                cv2.imwrite(img_path, final_img)

                # ---------------- plot gt token labels on the input image
                feat_img = gt_feat_map[k].permute(1, 2, 0).cpu().float().numpy() * 255
                img_path = os.path.join(out_dir, f'{img_id}_feat_img_final_{gt_name}.jpg')
                cv2.imwrite(img_path, feat_img)
                final_img = cv2.addWeighted(ori_img, 0.5, feat_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_final_img_final_{gt_name}_overlap.jpg')
                cv2.imwrite(img_path, final_img)

        # # -------------------- plot with the feature map size
        # h, w = list(self.sgdt_output['feat_map_size'].cpu().numpy())
        # sgdt_targets = resize_sgdt_target(self.sgdt_target_raw, feat_map_size=self.sgdt_output['feat_map_size'])
        # fg_gt, scale_gt = sgdt_targets['fg_gt'], sgdt_targets['scale_gt']
        # N, _ = fg_gt.shape

        img_resized = F.interpolate(img, [h, w], mode='bilinear').bool().float()
        for id, (gt, gt_name) in enumerate(zip([fg_gt, scale_gt], ['fg_gt', 'scale_gt'])):
            gt_feat_map = gt.permute(1, 0).view(-1, h, w)[:, None, :, :].repeat(1, 3, 1, 1)  # N,B -> B, 3, H, W,

            for k in range(B):
                img_id = self.targets[k]['image_id'].cpu().item()
                ori_img = img_resized[k].permute(1, 2, 0).cpu().numpy() * 255

                box_img = ori_img.copy()
                img_target = self.targets[k]
                # input_img_size = img_target['size']
                num_box = len(img_target['boxes'])
                box_unnormalized = unnormalize_box(box_normalized=img_target['boxes'],
                                                   input_img_size=img_target['size'])

                if num_box > 0:  # tensor([32.0000, 31.6452], device='cuda:0')
                    for box in box_unnormalized:
                        x1, y1, x2, y2 = box
                        box_img = cv2.rectangle(box_img, (int(x1), int(y1)), (int(x2), int(y2)), smrc.utils.YELLOW, 4)
                box_img = self.draw_token_edge(img=box_img, input_img_size=(H, W), feat_map_size=(h, w))

                img_path = os.path.join(out_dir, f'{img_id}_resized_box_img.jpg')
                cv2.imwrite(img_path, box_img)

                ori_img = box_img.copy()
                # feat_img = feat_map[k].permute(1, 2, 0).cpu().numpy() * 255
                feat_img = gt_feat_map[k].permute(1, 2, 0).cpu().float().numpy() * 255
                final_img = cv2.addWeighted(ori_img, 0.5, feat_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_resized_final_img_{gt_name}_overlap.jpg')
                cv2.imwrite(img_path, final_img)

                img_path = os.path.join(out_dir, f'{img_id}_resized_original_img.jpg')
                cv2.imwrite(img_path, ori_img)

                img_path = os.path.join(out_dir, f'{img_id}_resized_feat_img_{gt_name}.jpg')
                cv2.imwrite(img_path, feat_img)

    def visualize_prediction(self):
        """
            # ---------------- for visualization purpose
            fg_mask=adapted_token_dict['fg_mask'],
            fg_obj_score=adapted_token_dict['fg_obj_score'],
            tokens_to_discard_original=adapted_token_dict['tokens_to_discard_original'],
            small_scale_score=adapted_token_dict['small_scale_score'],
            scale_mask=adapted_token_dict['scale_mask'],
            tokens_to_split_original=adapted_token_dict['tokens_to_split_original'],
        Returns:

        """

        # adapted_pos = self.sgdt_output['adapted_pos']

        # fg_score = self.sgdt_output['fg_obj_score']  # B, N; torch.Size([2, 630])
        # small_scale_score = self.sgdt_output['small_scale_score']

        fg_score_logit = self.sgdt_output['fg_score_logit']  # N, B, Num_class
        small_scale_score_logit = self.sgdt_output['small_scale_score_logit']
        fg_labels = fg_score_logit.max(dim=-1).indices
        small_scale_labels = small_scale_score_logit.max(dim=-1).indices

        fg_mask = self.sgdt_output['fg_mask']  # N, B, Num_class
        scale_mask = self.sgdt_output['scale_mask']  # N, B, Num_class

        tokens_to_discard_original = self.sgdt_output['tokens_to_discard_original']
        tokens_to_split_original = self.sgdt_output['tokens_to_split_original']

        tokens_small_obj = self.sgdt_output['tokens_small_obj']
        tokens_to_discard = self.sgdt_output['tokens_to_discard']

        out_dir = self.out_dir

        # -------------------- plot on image of input size
        img = self.targets[-1]['input_imgs']
        B, C, H, W = img.shape  # (Tensor[B, 3, H, W])
        fg_gt_raw, scale_gt_raw = self.sgdt_target_raw['fg_gt'], self.sgdt_target_raw['scale_gt']

        # -------------------- plot on image of feature map size
        h, w = list(self.sgdt_output['feat_map_size'].detach().cpu().numpy())
        sgdt_targets = resize_sgdt_target(self.sgdt_target_raw, feat_map_size=self.sgdt_output['feat_map_size'])
        fg_gt, scale_gt = sgdt_targets['fg_gt'], sgdt_targets['scale_gt']
        N, _ = fg_gt.shape

        # -------------------------------------------------
        for i, (gt, gt_name, gt_on_feat_map, pred_labels, pred_mask,
                pred_sampled_token_label, pred_fixed_num_token_label) in enumerate(zip(
            [fg_gt_raw, scale_gt_raw],
            ['fg', 'scale'],
            [fg_gt, scale_gt],
            [fg_labels, small_scale_labels],
            [fg_mask, scale_mask],
            [tokens_to_discard_original, tokens_to_split_original],
            [tokens_to_discard, tokens_small_obj],
        )):
            gt_raw = gt[:, None, :, :].repeat(1, 3, 1, 1)  # B, H, W -> B, 3, H, W,

            gt_feat_map = gt_on_feat_map.permute(1, 0).view(-1, h, w)[:, None, :, :].repeat(
                1, 3, 1, 1)  # N,B -> B, 3, H, W,
            gt_feat_map = F.interpolate(gt_feat_map.float(), [H, W],
                                        mode='bilinear').bool().float()  # torch.Size([2, 3, 1105, 736])

            pred_labels = self.resize_token_labels_to_input_img(pred_labels, feat_map_size=(h, w),
                                                                input_img_size=(H, W))

            # -----------------------------
            pred_mask = self.resize_token_one_hot_label_to_input_img(
                pred_mask, feat_map_size=(h, w), input_img_size=(H, W))
            pred_sampled_token_label = self.resize_token_labels_to_input_img(
                pred_sampled_token_label, feat_map_size=(h, w), input_img_size=(H, W))
            pred_fixed_num_token_label = self.resize_token_labels_to_input_img(
                pred_fixed_num_token_label, feat_map_size=(h, w), input_img_size=(H, W))

            for k in range(B):
                img_id = self.targets[k]['image_id'].detach().cpu().item()
                ori_img = img[k].permute(1, 2, 0).detach().cpu().numpy() * 255

                box_img = ori_img.copy()
                img_target = self.targets[k]
                num_box = len(img_target['boxes'])
                box_unnormalized = unnormalize_box(box_normalized=img_target['boxes'],
                                                   input_img_size=img_target['size'])

                if num_box > 0:  # tensor([32.0000, 31.6452], device='cuda:0')
                    for box in box_unnormalized:
                        x1, y1, x2, y2 = box
                        box_img = cv2.rectangle(box_img, (int(x1), int(y1)), (int(x2), int(y2)), smrc.utils.YELLOW, 4)

                box_img = self.draw_token_edge(img=box_img, input_img_size=(H, W), feat_map_size=(h, w))

                img_path = os.path.join(out_dir, f'{img_id}_box_img.jpg')
                if not os.path.isfile(img_path):
                    cv2.imwrite(img_path, box_img)

                ori_img = box_img.copy()
                # img_path = os.path.join(out_dir, f'{img_id}_original_img.jpg')
                # if not os.path.isfile(img_path):
                #     cv2.imwrite(img_path, ori_img)

                # # feat_img = feat_map[k].permute(1, 2, 0).cpu().numpy() * 255
                # feat_img = gt_raw[k].permute(1, 2, 0).cpu().float().numpy() * 255
                # img_path = os.path.join(out_dir, f'{img_id}_feat_img_raw_{gt_name}.jpg')
                # cv2.imwrite(img_path, feat_img)
                # final_img = cv2.addWeighted(ori_img, 0.5, feat_img, 0.5, 2.2)
                # img_path = os.path.join(out_dir, f'{img_id}_final_img_raw_{gt_name}_overlap.jpg')
                # cv2.imwrite(img_path, final_img)

                # ---------------- plot gt token labels on the input image
                feat_img = self.resized_token_label_to_img_array(gt_feat_map[k])
                # img_path = os.path.join(out_dir, f'{img_id}_gt_{gt_name}.jpg')
                # cv2.imwrite(img_path, feat_img)
                final_img = cv2.addWeighted(ori_img, 0.5, feat_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_gt_{gt_name}_overlap.jpg')
                cv2.imwrite(img_path, final_img)

                # pred_img = pred_labels[k].permute(1, 2, 0).detach().cpu().float().numpy() * 255
                pred_img = self.resized_token_label_to_img_array(pred_labels[k])
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}.jpg')
                cv2.imwrite(img_path, pred_img)

                final_img = cv2.addWeighted(ori_img, 0.5, pred_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_overlap.jpg')
                cv2.imwrite(img_path, final_img)

                #
                # pred_img = pred_mask[k].permute(1, 2, 0).cpu().float().numpy() # * 255
                # pred_img = pred_img * np.array(smrc.utils.color.RED).reshape(3, 1, 1) + \
                #            (1 - pred_img) * np.array(smrc.utils.color.YELLOW).reshape(3, 1, 1)
                pred_img = self.resized_token_one_hot_label_to_color_img(pred_mask[k])
                # fg red, bg yellow.
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_mask.jpg')
                cv2.imwrite(img_path, pred_img)
                final_img = cv2.addWeighted(ori_img, 0.5, pred_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_mask_overlap.jpg')
                cv2.imwrite(img_path, final_img)
                # --------------
                pred_img = self.resized_token_label_to_img_array(pred_sampled_token_label[k])
                # self.token_mask_to_color_img(pred_sampled_token_label[k])
                # fg red, bg yellow.
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_pred_sampled_token_label.jpg')
                cv2.imwrite(img_path, pred_img)
                final_img = cv2.addWeighted(ori_img, 0.5, pred_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_pred_sampled_token_label_overlap.jpg')
                cv2.imwrite(img_path, final_img)

                pred_img = self.resized_token_label_to_img_array(pred_fixed_num_token_label[k])
                # fg red, bg yellow.
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_pred_fixed_num_token_label.jpg')
                cv2.imwrite(img_path, pred_img)
                final_img = cv2.addWeighted(ori_img, 0.5, pred_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_pred_fixed_num_token_label_overlap.jpg')
                cv2.imwrite(img_path, final_img)

        # # -------------------- plot with the feature map size
        # h, w = list(self.sgdt_output['feat_map_size'].cpu().numpy())
        # sgdt_targets = resize_sgdt_target(self.sgdt_target_raw, feat_map_size=self.sgdt_output['feat_map_size'])
        # fg_gt, scale_gt = sgdt_targets['fg_gt'], sgdt_targets['scale_gt']
        # N, _ = fg_gt.shape

        # img_resized = F.interpolate(img, [h, w], mode='nearest')
        # for i, (gt, gt_name) in enumerate(zip([fg_gt, scale_gt], ['fg_gt', 'scale_gt'])):
        #     gt_feat_map = gt.permute(1, 0).view(-1, h, w)[:, None, :, :].repeat(1, 3, 1, 1)  # N,B -> B, 3, H, W,
        #
        #     for k in range(B):
        #         img_id = self.targets[k]['image_id'].cpu().item()
        #         ori_img = img_resized[k].permute(1, 2, 0).cpu().numpy() * 255
        #
        #         box_img = ori_img.copy()
        #         img_target = self.targets[k]
        #         # input_img_size = img_target['size']
        #         num_box = len(img_target['boxes'])
        #         box_unnormalized = unnormalize_box(box_normalized=img_target['boxes'],
        #                                            input_img_size=img_target['size'])
        #
        #         if num_box > 0:  # tensor([32.0000, 31.6452], device='cuda:0')
        #             for box in box_unnormalized:
        #                 x1, y1, x2, y2 = box
        #                 box_img = cv2.rectangle(box_img, (int(x1), int(y1)), (int(x2), int(y2)), smrc.utils.YELLOW, 4)
        #         box_img = self.draw_token_edge(img=box_img, input_img_size=(H, W), feat_map_size=(h, w))

        #         img_path = os.path.join(out_dir, f'{img_id}_resized_box_img.jpg')
        #         cv2.imwrite(img_path, box_img)
        #
        #         ori_img = box_img.copy()
        #         # feat_img = feat_map[k].permute(1, 2, 0).cpu().numpy() * 255
        #         feat_img = gt_feat_map[k].permute(1, 2, 0).cpu().float().numpy() * 255
        #         final_img = cv2.addWeighted(ori_img, 0.5, feat_img, 0.5, 2.2)
        #         img_path = os.path.join(out_dir, f'{img_id}_resized_final_img_{gt_name}_overlap.jpg')
        #         cv2.imwrite(img_path, final_img)
        #
        #         img_path = os.path.join(out_dir, f'{img_id}_resized_original_img.jpg')
        #         cv2.imwrite(img_path, ori_img)
        #
        #         img_path = os.path.join(out_dir, f'{img_id}_resized_feat_img_{gt_name}.jpg')
        #         cv2.imwrite(img_path, feat_img)

    # deprecated.
    @staticmethod
    def token_score_feat_map_to_img(token_scores):
        final_img = np.float32(token_scores.permute(1, 2, 0).detach().cpu().float().numpy() * \
                               np.array([0.0, 0.0, 255.0]))
        return final_img

    # @staticmethod
    # def token_mask_to_color_img(pred_mask):
    #     """
    #             pred_img = pred_mask[k].permute(1, 2, 0).cpu().float().numpy()  # * 255
    #     pred_img = pred_img * np.array(smrc.utils.color.RED).reshape(3, 1, 1) + \
    #                (1 - pred_img) * np.array(smrc.utils.color.YELLOW).reshape(3, 1, 1)
    #     Args:
    #         pred_mask:
    #
    #     Returns:
    #
    #     """
    #     num_label = pred_mask.shape[-1]
    #     colors = smrc.utils.color.color_map
    #
    #     # 3, H, W, Num_label ->
    #     pred_img = pred_mask.permute(1, 2, 0).detach().cpu().float().numpy()  # * 255
    #     for k in range(num_label):
    #         mask = pred_img[:, :, -1] == k
    #         mask = np.expand_dims(mask, axis=0)
    #         pred_img += mask * colors[k].reshape(3, 1, 1)
    #     return pred_img
    #

    # def plot_heatmap(self, ):


def parser_encoder_layers(encoder_layer_config):
    layer_conf_split = encoder_layer_config.split('-')
    encoder_layer_list = []
    for l_conf in layer_conf_split:
        l_type_and_num = l_conf.split('_')
        assert len(l_type_and_num) == 2, f'The format of encoder layer config is wrong, ' \
                                         'expected length 2, e.g., regular_6, but got' \
                                         '{l_conf}'
        l_type, num_l = l_type_and_num[0], int(l_type_and_num[1])
        assert l_type in ['regular', 'sgdt', 'sgdtv1'] and num_l > 0
        encoder_layer_list.append([l_type, num_l])
    return encoder_layer_list


def get_num_sgdt_layer(encoder_layer_config):
    encoder_layer_list = parser_encoder_layers(encoder_layer_config)
    cnt = 0
    for (l_type, num_l) in encoder_layer_list:
        if l_type in ['sgdt', 'sgdtv1']:
            cnt += num_l
    return cnt


def sgdt_update_sample_target(samples, targets, args):
    # ------------------- tti
    b, c, h, w = samples.tensors.shape  # b, c, h, w
    for i, target in enumerate(targets):  # a list
        target['padded_img_size'] = torch.as_tensor([int(h), int(w)])
        # if args.token_adaption_visualization:  # save the input image into targets also for visualization
        #     tensor_i = samples.tensors[i]
        #     mask_i = samples.mask[i]
        #     target['input_img'] = samples.to_img_list_single(tensor=tensor_i, mask=mask_i)
    mean = samples.tensors.new_tensor([0.485, 0.456, 0.406])
    std = samples.tensors.new_tensor([0.229, 0.224, 0.225])  #
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    if args.token_adaption_visualization:
        targets[-1]['input_imgs'] = samples.tensors * std[None, :, None, None] + mean[None, :, None, None]
    # ------------------- tti
    return samples, targets


def update_targets_with_proposals(proposals, targets):
    """

    Args:
        proposals: [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        targets: a list of dict, each of them is
            'boxes' = {Tensor: 20} tensor([[0.3896, 0.4161, 0.0386, 0.1631],\n        [0.1276, 0.5052, 0.2333, 0.2227],\n        [0.9342, 0.5835, 0.1271, 0.1848],\n        [0.6047, 0.6325, 0.0875, 0.2414],\n        [0.5025, 0.6273, 0.0966, 0.2312],\n        [0.6692, 0.6190, 0.0471, 0.1910],\n        [0.5128, 0.5283, 0.0337, 0.0272],\n        [0.6864, 0.5320, 0.0829, 0.3240],\n        [0.6125, 0.4462, 0.0236, 0.0839],\n        [0.8119, 0.5017, 0.0230, 0.0375],\n        [0.7863, 0.5364, 0.0317, 0.2542],\n        [0.9562, 0.7717, 0.0224, 0.1073],\n        [0.9682, 0.7781, 0.0201, 0.1090],\n        [0.7106, 0.3100, 0.0218, 0.0514],\n        [0.8866, 0.8316, 0.0573, 0.2105],\n        [0.5569, 0.5167, 0.0178, 0.0529],\n        [0.6517, 0.5288, 0.0150, 0.0294],\n        [0.3880, 0.4784, 0.0222, 0.0414],\n        [0.5338, 0.4879, 0.0152, 0.0393],\n        [0.6000, 0.6471, 0.1962, 0.2088]], device='cuda:0')
            'labels' = {Tensor: 20} tensor([64, 72, 72, 62, 62, 62, 62,  1,  1, 78, 82, 84, 84, 85, 86, 86, 62, 86,\n        86, 67], device='cuda:0')
            'image_id' = {Tensor: 1} tensor([139], device='cuda:0')
            'area' = {Tensor: 20} tensor([ 1874.1207, 46674.9805, 20556.2637,  7912.7275,  6462.3667,  4543.8306,\n          740.5751, 10265.9785,  1533.4774,   767.2557,  7365.1987,  1193.2782,\n         1136.8395,   795.2544,  7652.9175,   627.9352,   320.6446,   668.0214,\n          423.7049,  8325.5576], device='cuda:0')
            'iscrowd' = {Tensor: 20} tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n       device='cuda:0')
            'orig_size' = {Tensor: 2} tensor([426, 640], device='cuda:0')
            'size' = {Tensor: 2} tensor([ 800, 1201], device='cuda:0')
            'original_area' = {Tensor: 20} tensor([  531.8071, 13244.6572,  5833.1182,  2245.3435,  1833.7841,  1289.3734,\n          210.1482,  2913.1104,   435.1450,   217.7192,  2089.9749,   338.6089,\n          322.5936,   225.6642,  2171.6189,   178.1851,    90.9873,   189.5601,\n          120.23
            'padded_img_size' = {Tensor: 2} tensor([ 873, 1201], device='cuda:0')

    Returns:

    """
    #

    for k in range(len(proposals)):
        # updated_targets[k] = {}
        targets[k]['proposal_boxes'] = proposals[k]['boxes']
        targets[k]['proposal_labels'] = proposals[k]['labels']
        targets[k]['proposal_scores'] = proposals[k]['scores']

    return targets


def extract_proposals_as_targets(targets):
    # targets is a list
    targets_proposal = copy.deepcopy(targets)
    for k in range(len(targets)):
        # updated_targets[k] = {}
        targets_proposal[k]['boxes'] = targets[k]['proposal_boxes']
        targets_proposal[k]['labels'] = targets[k]['proposal_labels']
        targets_proposal[k]['scores'] = targets[k]['proposal_scores']

    return targets_proposal


def get_targets_from_proposals(proposals, targets):
    """

    Args:
        proposals: [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        targets: a list of dict, each of them is
            'boxes' = {Tensor: 20} tensor([[0.3896, 0.4161, 0.0386, 0.1631],\n        [0.1276, 0.5052, 0.2333, 0.2227],\n        [0.9342, 0.5835, 0.1271, 0.1848],\n        [0.6047, 0.6325, 0.0875, 0.2414],\n        [0.5025, 0.6273, 0.0966, 0.2312],\n        [0.6692, 0.6190, 0.0471, 0.1910],\n        [0.5128, 0.5283, 0.0337, 0.0272],\n        [0.6864, 0.5320, 0.0829, 0.3240],\n        [0.6125, 0.4462, 0.0236, 0.0839],\n        [0.8119, 0.5017, 0.0230, 0.0375],\n        [0.7863, 0.5364, 0.0317, 0.2542],\n        [0.9562, 0.7717, 0.0224, 0.1073],\n        [0.9682, 0.7781, 0.0201, 0.1090],\n        [0.7106, 0.3100, 0.0218, 0.0514],\n        [0.8866, 0.8316, 0.0573, 0.2105],\n        [0.5569, 0.5167, 0.0178, 0.0529],\n        [0.6517, 0.5288, 0.0150, 0.0294],\n        [0.3880, 0.4784, 0.0222, 0.0414],\n        [0.5338, 0.4879, 0.0152, 0.0393],\n        [0.6000, 0.6471, 0.1962, 0.2088]], device='cuda:0')
            'labels' = {Tensor: 20} tensor([64, 72, 72, 62, 62, 62, 62,  1,  1, 78, 82, 84, 84, 85, 86, 86, 62, 86,\n        86, 67], device='cuda:0')
            'image_id' = {Tensor: 1} tensor([139], device='cuda:0')
            'area' = {Tensor: 20} tensor([ 1874.1207, 46674.9805, 20556.2637,  7912.7275,  6462.3667,  4543.8306,\n          740.5751, 10265.9785,  1533.4774,   767.2557,  7365.1987,  1193.2782,\n         1136.8395,   795.2544,  7652.9175,   627.9352,   320.6446,   668.0214,\n          423.7049,  8325.5576], device='cuda:0')
            'iscrowd' = {Tensor: 20} tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n       device='cuda:0')
            'orig_size' = {Tensor: 2} tensor([426, 640], device='cuda:0')
            'size' = {Tensor: 2} tensor([ 800, 1201], device='cuda:0')
            'original_area' = {Tensor: 20} tensor([  531.8071, 13244.6572,  5833.1182,  2245.3435,  1833.7841,  1289.3734,\n          210.1482,  2913.1104,   435.1450,   217.7192,  2089.9749,   338.6089,\n          322.5936,   225.6642,  2171.6189,   178.1851,    90.9873,   189.5601,\n          120.23
            'padded_img_size' = {Tensor: 2} tensor([ 873, 1201], device='cuda:0')

    Returns:

    """
    #
    updated_targets = targets.copy()
    for k in range(len(targets)):
        # updated_targets[k] = {}
        updated_targets[k]['boxes'] = proposals[k]['boxes']
        updated_targets[k]['labels'] = proposals[k]['labels']
        updated_targets[k]['scores'] = proposals[k]['scores']

    return updated_targets
