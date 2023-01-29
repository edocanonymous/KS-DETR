import math
import torch
import numpy as np
import torch.nn.functional as F


# from .scoring_interpolate import interpolate_modified # we modified its adaptive pool to max pool  area


# areaRngSGDT = [0 ** 2, 32 ** 2, 96 ** 2, 1e5 ** 2]  # defined in the original image space.
areaRngSGDT = [0 ** 2, 32 ** 2, 96 ** 2, 128 ** 2, 256 ** 2, 1e5 ** 2]  # Here we define in the input image space.

# coco definition
areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
areaRngLbl = ['all', 'small', 'medium', 'large']

"""https://discuss.pytorch.org/t/downsampling-tensors/16617/4
Upsample uses F.interpolate as suggested.
We can check the source to see what’s actually doing: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#interpolate 2.2k
The default mode ‘nearest’ doesn’t suit downscaling but ‘bilinear’ works fine (please correct me if I’m wrong!).
Otherwise, just use mode ‘area’ or adaptive_avg_pool2d directly. However, the latter seems a bit slower than ‘bilinear’.
"""
# interpolate_mode='nearest'  # nearest   bilinear area
INTERPOLATE_MODE = 'bilinear'  # Never changing it back to nearest mode for 0, 1 downsamling interpolation


def is_small_medium_object(obj_area):
    # we change to the coco definition
    # small + medium: areaRngSGDT[3] if we use the resized 'area' instead of the 'original area'
    if obj_area < areaRngSGDT[2]:
        return True
    else:
        return False


def is_small_object(obj_area):
    if obj_area < areaRngSGDT[1]:  # small areaRngSGDT[2]
        return True
    else:
        return False


def unnormalize_box(box_normalized, input_img_size):
    """
            # ksgt_output['feat_map_size'] = torch.tensor([h, w])
        # feat_map_size = ksgt_output['feat_map_size']  # [h, w] tensor([23, 31], device='cuda:0')

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


def prepare_ksgt_targets_v0(targets, pad_fg_pixel=16, token_scoring_gt_criterion=None):
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
           ksgt_targets_raw:  # cannot be used as gt.
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
    padded_img_size = tuple(targets[0]['padded_img_size'].cpu().numpy())  # (736, 981)
    batch_size = len(targets)

    mask_size = (batch_size,) + padded_img_size

    # We must not use scale_gt = fg_gt = torch.zeros(), otherwise, they will share the same variable.
    fg_gt = torch.zeros(mask_size).to(targets[0]['size'].device).float()  # H, W
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

            # target['original_area'] = target['area'].clone()
            if token_scoring_gt_criterion == 'significance_value_from_instance_mask':
                assert 'masks' in img_target and len(box_unnormalized) == len(img_target['masks'])

            # from large box to small box to ensure smaller box always have a large significant value than
            # large box.
            assert len(box_unnormalized) == len(img_target['original_area'])
            # inds = torch.argsort(img_target['original_area'], descending=True)
            # for kk in inds:
            #     box = box_unnormalized[kk]
            #     box_area = img_target['original_area'][kk]
            for kk, (box, box_area) in enumerate(
                    zip(box_unnormalized, img_target['original_area'])):  # ('original_area', 'area')
                x1, y1, x2, y2 = box

                # we use the recalculated box_are instead of the saved area because we may use the area of proposal
                # in that case, we cannot pre-save the area.
                # box_area = (x2 - x1) * (y2 - y1)
                # assert box_area >= 0

                fg_gt[k, y1:y2, x1:x2] = 1.0  # foreground objects

                # soft label for small object so that smaller objects has large value (e.g., 1 vs. 0.9)
                #  than relative large small objects
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

                    elif token_scoring_gt_criterion == 'significance_value_from_instance_mask':
                        instance_mask = img_target['masks'][kk]
                        # # the regions to update is not a rectangle any more, but a binary mask.
                        # instance_mask_h, instance_mask_w = instance_mask.shape
                        # instance_mask_padded = torch.full(padded_img_size, False, device=instance_mask.device)
                        # instance_mask_padded[:instance_mask_h, :instance_mask_w] = instance_mask

                        padding_bottom, padding_right = padded_img_size[0] - instance_mask.shape[0], \
                                                        padded_img_size[1] - instance_mask.shape[1]
                        m = torch.nn.ZeroPad2d((0, padding_right, 0, padding_bottom))
                        instance_mask_padded = m(instance_mask.float().unsqueeze(0)).bool().squeeze(0)
                        # assert torch.equal(instance_mask_padded, instance_mask_padded1)

                        scale_gt[k][instance_mask_padded] = torch.max(
                            scale_gt[k][instance_mask_padded],
                            scale_gt.new_tensor(significant_score)
                        )
                    else:
                        # Use the max significant_score if overlap exists
                        scale_gt[k, y1:y2, x1:x2] = torch.max(
                            scale_gt[k, y1:y2, x1:x2],
                            scale_gt.new_tensor(significant_score)
                        )
                elif token_scoring_gt_criterion in ['fg_scale_class_all_fg',  'fg_scale_class_all_bg']:
                    scale_gt[k, y1:y2, x1:x2] = 1.0
                elif token_scoring_gt_criterion == 'fg_scale_class_small_medium_random':
                    # TODO: soft label for small object so that smaller objects has large value (e.g., 1 vs. 0.9)
                    #  than relative large small objects
                    if is_small_medium_object(box_area):  # small object or not
                        scale_gt[k, y1:y2, x1:x2] = 1.0
                elif token_scoring_gt_criterion == 'fg_scale_class_small_random':
                    if is_small_object(box_area):  # small object or not
                        scale_gt[k, y1:y2, x1:x2] = 1.0
                else:
                    raise NotImplementedError

    # fg_gt = torch.where(scale_gt.float() > 0, 1.0, 0.0)

    # set all bg tokens to 1.0
    if token_scoring_gt_criterion == 'fg_scale_class_all_bg':
        scale_gt = 1 - scale_gt
        fg_gt = 1 - fg_gt

    ksgt_targets_raw = dict(
        fg_gt=fg_gt.float(),  # B, H, W
        scale_gt=scale_gt.float()  # B, H, W
    )

    return ksgt_targets_raw


def prepare_ksgt_targets(targets, padded_img_size, pad_fg_pixel=16,  token_scoring_gt_criterion=None):
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
           ksgt_targets_raw:  # cannot be used as gt.
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
    # padded_img_size = tuple(targets[0]['padded_img_size'].cpu().numpy())  # (736, 981)

    batch_size = len(targets)
    mask_size = (batch_size,) + padded_img_size

    # We must not use scale_gt = fg_gt = torch.zeros(), otherwise, they will share the same variable.
    fg_gt = torch.zeros(mask_size).to(targets[0]['boxes'].device).float()  # H, W
    scale_gt = torch.zeros(mask_size).to(targets[0]['boxes'].device).float()

    # padded_img_area = torch.prod(padded_img_size)
    for k, img_target in enumerate(targets):
        if token_scoring_gt_criterion == 'fake_all_tokens_are_fg':
            scale_gt = torch.ones_like(scale_gt)
            fg_gt = torch.ones_like(fg_gt)
            continue

        # # 0 means bg, 1, fg. -1 means padding position.
        # box_unnormalized = unnormalize_box(box_normalized=img_target['boxes'],
        #                                    input_img_size=img_target['size'])
        box_unnormalized = img_target['box_unnormalized'].type(torch.int64)
        num_box = len(img_target['boxes'])
        if num_box > 0:  # tensor([32.0000, 31.6452], device='cuda:0')
            # ------------------------- Extend the fg regions
            if pad_fg_pixel > 0:
                input_img_size = img_target['image_size']  # input image size
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

            # target['original_area'] = target['area'].clone()
            if token_scoring_gt_criterion == 'significance_value_from_instance_mask':
                assert 'masks' in img_target and len(box_unnormalized) == len(img_target['masks'])

            for kk, box in enumerate(box_unnormalized):  # ('original_area', 'area')
                x1, y1, x2, y2 = box

                # we use the recalculated box_are instead of the saved area because we may use the area of proposal
                # in that case, we cannot pre-save the area.
                box_area = (x2 - x1) * (y2 - y1)
                # assert box_area >= 0

                fg_gt[k, y1:y2, x1:x2] = 1.0  # foreground objects

                # soft label for small object so that smaller objects has large value (e.g., 1 vs. 0.9)
                #  than relative large small objects
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

                    elif token_scoring_gt_criterion == 'significance_value_from_instance_mask':
                        instance_mask = img_target['masks'][kk]
                        # # the regions to update is not a rectangle any more, but a binary mask.
                        # instance_mask_h, instance_mask_w = instance_mask.shape
                        # instance_mask_padded = torch.full(padded_img_size, False, device=instance_mask.device)
                        # instance_mask_padded[:instance_mask_h, :instance_mask_w] = instance_mask

                        padding_bottom, padding_right = padded_img_size[0] - instance_mask.shape[0], \
                                                        padded_img_size[1] - instance_mask.shape[1]
                        m = torch.nn.ZeroPad2d((0, padding_right, 0, padding_bottom))
                        instance_mask_padded = m(instance_mask.float().unsqueeze(0)).bool().squeeze(0)
                        # assert torch.equal(instance_mask_padded, instance_mask_padded1)

                        scale_gt[k][instance_mask_padded] = torch.max(
                            scale_gt[k][instance_mask_padded],
                            scale_gt.new_tensor(significant_score)
                        )
                    else:
                        # Use the max significant_score if overlap exists
                        scale_gt[k, y1:y2, x1:x2] = torch.max(
                            scale_gt[k, y1:y2, x1:x2],
                            scale_gt.new_tensor(significant_score)
                        )
                elif token_scoring_gt_criterion in ['fg_scale_class_all_fg',  'fg_scale_class_all_bg']:
                    scale_gt[k, y1:y2, x1:x2] = 1.0
                elif token_scoring_gt_criterion == 'fg_scale_class_small_medium_random':
                    # TODO: soft label for small object so that smaller objects has large value (e.g., 1 vs. 0.9)
                    #  than relative large small objects
                    if is_small_medium_object(box_area):  # small object or not
                        scale_gt[k, y1:y2, x1:x2] = 1.0
                elif token_scoring_gt_criterion == 'fg_scale_class_small_random':
                    if is_small_object(box_area):  # small object or not
                        scale_gt[k, y1:y2, x1:x2] = 1.0
                else:
                    raise NotImplementedError

    # fg_gt = torch.where(scale_gt.float() > 0, 1.0, 0.0)

    # set all bg tokens to 1.0
    if token_scoring_gt_criterion == 'fg_scale_class_all_bg':
        scale_gt = 1 - scale_gt
        fg_gt = 1 - fg_gt

    ksgt_targets_raw = dict(
        fg_gt=fg_gt.float(),  # B, H, W
        scale_gt=scale_gt.float()  # B, H, W
    )

    return ksgt_targets_raw


def resize_ksgt_target(ksgt_targets, feat_map_size,
                       # feat_map_mask=None,
                       interpolate_mode=INTERPOLATE_MODE  # nearest   bilinear
                       ):
    """
    F.interpolate default mode is 'nearest'

    UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0.
    Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details.
    Args:
        interpolate_mode:
        ksgt_targets:
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
    # if the input can be divided into
    # output_size = (28, 38)
    # input_size = ksgt_targets['scale_gt'].shape  # torch.Size([2, 873, 1201])
    # H, W = input_size[-2:]  # 873, 1201
    # h, w = output_size[0], output_size[1]
    # stride_h, stride_w = int(np.ceil(H / h)), int(np.ceil(W / w))
    # # 1) a single int – in which case the same value is used for the height and width dimension
    # # 2) a tuple of two ints – in which case, the first int is used
    # # for the height dimension, and the second int for the width dimension
    # max_pool = torch.nn.MaxPool2d(kernel_size=(stride_h, stride_w), stride=(stride_h, stride_w),
    #                               ceil_mode=True)
    ksgt_targets_resized = {}
    for k, gt in ksgt_targets.items():  # gt could be fg_gt, scale_gt, proposal_fg_gt, proposal_scale_gt
        gt_binary = True if gt.unique().shape[0] == 2 else False
        gt_new = F.interpolate(gt[None].float(), size=output_size, mode=interpolate_mode, align_corners=True)[0]
        # gt_new = interpolate_modified(gt[None].float(), size=output_size, mode=interpolate_mode,
        # align_corners=True)[0]

        # input (N,C,H,W) or (C,H,W), output (N, C, H_{out}, W_{out}) or (C, H_{out}, W_{out})
        # gt_new = max_pool(gt[None].float(), size=output_size, mode=interpolate_mode)

        if gt_binary:
            gt_new = gt_new.bool().float()

        ksgt_targets_resized[k] = gt_new.flatten(1).permute(1, 0)  # float -> long  .long()

    return ksgt_targets_resized  # torch.Size([600, 2]), (N, B)


class TokenScoringGTGenerator:
    def __init__(self,
                 token_scoring_gt_criterion,
                 pad_fg_pixel=None,
                 # proposal_scoring=None,
                 # proposal_token_scoring_gt_criterion=None,
                 ):
        self.token_scoring_gt_criterion = token_scoring_gt_criterion

        self.pad_fg_pixel = pad_fg_pixel
        self.sig_value_interpolate_mode = INTERPOLATE_MODE  # nearest cause round off error

        # self.proposal_token_scoring_gt_criterion = proposal_token_scoring_gt_criterion
        # self.proposal_scoring = proposal_scoring
        # self.proposal_scoring_parser = None if proposal_scoring is None else \
        #     ProposalScoringParser(proposal_scoring_config=proposal_scoring)

    def get_gt_raw(self, targets, padded_img_size):
        """
        Args:
            padded_img_size:
            targets: a list of dict
        Returns:
        """
        ksgt_targets_raw = prepare_ksgt_targets(
            targets=targets, pad_fg_pixel=self.pad_fg_pixel, padded_img_size=padded_img_size,
            token_scoring_gt_criterion=self.token_scoring_gt_criterion)
        return ksgt_targets_raw

    # def update_proposal_gt_raw(self, targets, selected_proposals, ksgt_target_raw: dict = None, **kwargs):  #
    #     targets_updated = update_targets_with_proposals(selected_proposals, targets)
    #
    #     if ksgt_target_raw is None:
    #         ksgt_target_raw = dict()
    #
    #     for t in targets_updated:
    #         assert 'proposal_boxes' in t
    #
    #     # extract proposals as targets
    #     # targets_proposal = extract_proposals_as_targets(targets)
    #
    #     if self.proposal_scoring_parser is not None:
    #         proposal_filtering_param = self.proposal_scoring_parser.extract_gt_split_remove_parameter()
    #     else:
    #         proposal_filtering_param = kwargs
    #
    #     # use the same setting of the gt targets if they are not set.
    #     if proposal_filtering_param.pop('pad_fg_pixel', None) is None:
    #         proposal_filtering_param['pad_fg_pixel'] = self.pad_fg_pixel
    #
    #     # if proposal_filtering_param.pop('proposal_token_scoring_gt_criterion', None) is None:
    #     #     proposal_filtering_param['proposal_token_scoring_gt_criterion'] = self.token_scoring_gt_criterion
    #     proposal_filtering_param['proposal_token_scoring_gt_criterion'] = self.proposal_token_scoring_gt_criterion
    #     if self.proposal_token_scoring_gt_criterion is None:
    #         proposal_filtering_param['proposal_token_scoring_gt_criterion'] = self.token_scoring_gt_criterion
    #
    #     ksgt_targets_proposal_raw = prepare_ksgt_proposal_significant_value(
    #         proposals=targets_updated, **proposal_filtering_param,
    #         # use_conf_score=use_conf_score,
    #         # min_fg_score=min_fg_score,
    #         # min_split_score=min_split_score,
    #         # pad_fg_pixel=pad_fg_pixel,
    #         # token_scoring_gt_criterion=token_scoring_gt_criterion,
    #     )
    #     ksgt_target_raw.update(ksgt_targets_proposal_raw)
    #
    #     return targets_updated, ksgt_target_raw

    def resize_sig_value_gt(self, ksgt_targets, feat_map_size):
        return resize_ksgt_target(
            ksgt_targets=ksgt_targets,
            feat_map_size=feat_map_size,
            interpolate_mode=self.sig_value_interpolate_mode
        )

    # def get_gt_raw_v0(self, targets, **kwargs):
    #     """
    #
    #     Args:
    #         targets: a list of dict
    #
    #     Returns:
    #
    #     """
    #     ksgt_targets_raw = prepare_ksgt_targets(
    #         targets=targets, pad_fg_pixel=self.pad_fg_pixel,
    #         token_scoring_gt_criterion=self.token_scoring_gt_criterion)
    #
    #     if 'proposal_boxes' in targets[0]:
    #         # extract proposals as targets
    #         targets_proposal = extract_proposals_as_targets(targets)
    #
    #         if self.proposal_scoring_parser is not None:
    #
    #             min_score = self.proposal_scoring_parser.extract_thd('min_score')
    #             nms_thd = self.proposal_scoring_parser.extract_thd('nms_thd')
    #             num_select = self.proposal_scoring_parser.extract_thd('num_select')
    #
    #             min_fg_score = self.proposal_scoring_parser.extract_thd('min_fg_score')
    #             min_split_score = self.proposal_scoring_parser.extract_thd('min_split_score')
    #
    #             use_conf_score = self.proposal_scoring_parser.str_exist('use_conf_score')
    #             # pad_fg_pixel = self.proposal_scoring_parser.extract_thd('pad_fg_pixel')
    #             # token_scoring_gt_criterion = self.proposal_scoring_parser.extract_sub_setting(
    #             #     'token_scoring_gt_criterion')
    #
    #             if self.proposal_scoring_parser.str_exist('pad_fg_pixel'):
    #                 pad_fg_pixel = self.proposal_scoring_parser.extract_thd('pad_fg_pixel')
    #             else:
    #                 pad_fg_pixel = self.pad_fg_pixel
    #
    #             if self.proposal_scoring_parser.str_exist('token_scoring_gt_criterion'):
    #                 token_scoring_gt_criterion = self.proposal_scoring_parser.extract_sub_setting(
    #                     'token_scoring_gt_criterion')
    #             else:
    #                 token_scoring_gt_criterion = self.token_scoring_gt_criterion
    #
    #         else:
    #             min_score = kwargs.pop('min_score', None)
    #             nms_thd = kwargs.pop('nms_thd', None)
    #             num_select = kwargs.pop('num_select', None)
    #
    #             min_fg_score = kwargs.pop('min_fg_score', None)
    #             min_split_score = kwargs.pop('min_split_score', None)
    #
    #             use_conf_score = kwargs.pop('use_conf_score', False)
    #             # pad_fg_pixel = kwargs.pop('pad_fg_pixel', None)
    #             # token_scoring_gt_criterion = kwargs.pop('token_scoring_gt_criterion', None)
    #             pad_fg_pixel = kwargs.pop('pad_fg_pixel', self.pad_fg_pixel)
    #             token_scoring_gt_criterion = kwargs.pop('token_scoring_gt_criterion', self.token_scoring_gt_criterion)
    #
    #         if min_score is not None or nms_thd is not None or num_select is not None:
    #             proposal_processor = ProposalProcess(
    #                 min_score=min_score,
    #                 nms_thd=nms_thd,
    #                 num_select=num_select)
    #             # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    #             targets_proposal_new = proposal_processor.bbox_filtering(
    #                 boxes=[x['boxes'] for x in targets_proposal],
    #                 scores=[x['scores'] for x in targets_proposal],
    #                 labels=[x['labels'] for x in targets_proposal],
    #             )
    #             # list of dict for both targets_proposal and targets_proposal_new
    #             for k in range(len(targets_proposal)):
    #                 targets_proposal[k].update(targets_proposal_new[k])
    #
    #         # generate gt from proposals.
    #         ksgt_targets_proposal_raw = prepare_ksgt_proposal_significant_value(
    #             proposals=targets_proposal,
    #             proposal_token_scoring_gt_criterion=token_scoring_gt_criterion,
    #             min_fg_score=min_fg_score,
    #             min_split_score=min_split_score,
    #             pad_fg_pixel=pad_fg_pixel,
    #             use_conf_score=use_conf_score,
    #         )
    #         ksgt_targets_raw['proposal_fg_gt'] = ksgt_targets_proposal_raw['fg_gt']
    #         ksgt_targets_raw['proposal_scale_gt'] = ksgt_targets_proposal_raw['scale_gt']
    #
    #     return ksgt_targets_raw


def ksgt_update_sample_target(samples, targets, args):
    # ------------------- tti
    b, c, h, w = samples.tensors.shape  # b, c, h, w
    for i, target in enumerate(targets):  # a list
        target['padded_img_size'] = torch.as_tensor([int(h), int(w)])

    mean = samples.tensors.new_tensor([0.485, 0.456, 0.406])
    std = samples.tensors.new_tensor([0.229, 0.224, 0.225])  #
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    if args.token_adaption_visualization:
        targets[-1]['input_imgs'] = samples.tensors * std[None, :, None, None] + mean[None, :, None, None]

    return samples, targets


