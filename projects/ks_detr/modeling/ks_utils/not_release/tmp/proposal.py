import copy

import torch

from projects.ks_detr.modeling import SGDTConfigParse
from projects.ks_detr.modeling.ks_utils.gt_mask_generator import estimate_sig_score_piecewise_linear, MIN_FG_SIGNIFICANCE, \
    is_small_medium_object, is_small_object


def prepare_ksgt_proposal_significant_value(
        proposals, pad_fg_pixel=None, proposal_token_scoring_gt_criterion=None,
        min_fg_score=None, min_split_score=None,
        use_conf_score=False,
):
    if proposal_token_scoring_gt_criterion is None:
        proposal_token_scoring_gt_criterion = 'significance_value'

    if min_fg_score is None:
        min_fg_score = 0.0

    if min_split_score is None:
        min_split_score = 0.0

    if pad_fg_pixel is None:
        pad_fg_pixel = 16

    if proposal_token_scoring_gt_criterion == 'confidence_value':
        use_conf_score = True

    # B, H, W
    padded_img_size = proposals[0]['padded_img_size']  # (736, 981)
    batch_size = len(proposals)
    mask_size = (batch_size,) + tuple(padded_img_size.cpu().numpy())

    # We must not use scale_gt = fg_gt = torch.zeros(), otherwise, they will share the same variable.
    proposal_fg_gt = torch.zeros(mask_size).to(proposals[0]['size'].device).float()  # H, W  TODO
    proposal_scale_gt = torch.zeros(mask_size).to(proposals[0]['size'].device).float()

    # padded_img_area = torch.prod(padded_img_size)
    for k, img_target in enumerate(proposals):
        if proposal_token_scoring_gt_criterion == 'fake_all_tokens_are_fg':
            proposal_scale_gt = torch.ones_like(proposal_scale_gt)
            proposal_fg_gt = torch.ones_like(proposal_fg_gt)
            continue

        # # # # 0 means bg, 1, fg. -1 means padding position.
        # box_unnormalized = unnormalize_box(box_normalized=img_target['proposal_boxes'],
        #                                    input_img_size=img_target['size'])

        # # ==================================================
        # # the proposals are already in the image coordinate system, no need to map them back.
        box_unnormalized = img_target['proposal_boxes'].int()
        # # ==================================================

        num_box = len(img_target['proposal_boxes'])
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
            # the area information is not available in the proposals.

            aspect_ratio = (img_target['orig_size'] / img_target['size']).prod()
            for kk, box in enumerate(box_unnormalized):
                # we use the recalculated box_are instead of the saved area because we may use the area of proposal
                # in that case, we cannot pre-save the area.
                x1, y1, x2, y2 = box
                # box_area = (x2 - x1) * (y2 - y1)  # the area on the image

                box_area = (x2 - x1) * (y2 - y1) * aspect_ratio  # the area on the orignal image
                assert box_area >= 0

                assert 'proposal_scores' in img_target
                conf_score = img_target['proposal_scores'][kk] if use_conf_score else 1.0
                if 'proposal_scores' in img_target and img_target['proposal_scores'][kk] < min_fg_score:
                    pass  # if score is too lower, ignore this for remove judgement.
                else:
                    proposal_fg_gt[k, y1:y2, x1:x2] = 1.0  # foreground objects

                if 'proposal_scores' in img_target and img_target['proposal_scores'][kk] < min_split_score:
                    pass  # skip generating the significant value for this box.
                else:
                    if proposal_token_scoring_gt_criterion == 'confidence_value':
                        # use the confidence value as significance and use the max value if overlap exists
                        proposal_scale_gt[k, y1:y2, x1:x2] = torch.max(
                            proposal_scale_gt[k, y1:y2, x1:x2],
                            proposal_scale_gt.new_tensor(conf_score)
                        )
                    elif proposal_token_scoring_gt_criterion.find('significance_value') > -1:
                        # significance_value_bg_w_priority

                        # significant_score = estimate_significant_score(box_area)
                        significant_score = estimate_sig_score_piecewise_linear(box_area)

                        if proposal_token_scoring_gt_criterion == 'significance_value_inverse_fg':
                            # inverse the significance of fg objects, so that larger has higher significance value.
                            # 1 -> MIN_FG_SIGNIFICANCE, MIN_FG_SIGNIFICANCE -> 1
                            significant_score = (1 - significant_score + MIN_FG_SIGNIFICANCE) * conf_score

                            fg_loc = proposal_scale_gt[k, y1:y2, x1:x2] > 0
                            bg_loc = proposal_scale_gt[k, y1:y2, x1:x2] <= 0
                            # Use the min significant_score if overlap exists so that the clutter regions has
                            # low priority to be sampled (only for debugging)
                            proposal_scale_gt[k, y1:y2, x1:x2][fg_loc] = torch.min(
                                proposal_scale_gt[k, y1:y2, x1:x2][fg_loc],
                                proposal_scale_gt.new_tensor(significant_score * conf_score)
                            )
                            # for bg locations, just update to significant_score
                            proposal_scale_gt[k, y1:y2, x1:x2][bg_loc] = proposal_scale_gt.new_tensor(
                                significant_score * conf_score)
                        else:
                            # Use the max significant_score if overlap exists
                            proposal_scale_gt[k, y1:y2, x1:x2] = torch.max(
                                proposal_scale_gt[k, y1:y2, x1:x2],
                                proposal_scale_gt.new_tensor(significant_score * conf_score)
                            )
                    elif proposal_token_scoring_gt_criterion == 'fg_scale_class_all_fg':
                        proposal_scale_gt[k, y1:y2, x1:x2] = 1.0 * conf_score
                    elif proposal_token_scoring_gt_criterion == 'fg_scale_class_small_medium_random':
                        # TODO: soft label for small object so that smaller objects has large value (e.g., 1 vs. 0.9)
                        #  than relative large small objects
                        if is_small_medium_object(box_area):  # small object or not
                            proposal_scale_gt[k, y1:y2, x1:x2] = 1.0 * conf_score
                    elif proposal_token_scoring_gt_criterion == 'fg_scale_class_small_random':
                        if is_small_object(box_area):  # small object or not
                            proposal_scale_gt[k, y1:y2, x1:x2] = 1.0 * conf_score
                    else:
                        raise NotImplementedError
                    # else:
                    #     print(f' False, {int(box_area)}  > {areaRngSGDT[2]}')

    ksgt_targets_raw = dict(
        proposal_fg_gt=proposal_fg_gt.float(),  # B, H, W
        proposal_scale_gt=proposal_scale_gt.float()  # B, H, W
    )
    return ksgt_targets_raw


class ProposalScoringParser(SGDTConfigParse):
    def __init__(self, proposal_scoring_config):
        assert proposal_scoring_config is not None
        super().__init__(config_str=proposal_scoring_config)

    def extract_box_filtering_parameter(self):
        min_score = self.extract_thd('min_score')
        nms_thd = self.extract_thd('nms_thd')
        num_select = self.extract_thd('num_select')

        return dict(min_score=min_score, nms_thd=nms_thd, num_select=num_select)

    def extract_gt_split_remove_parameter(self):
        min_fg_score = self.extract_thd('min_fg_score')
        min_split_score = self.extract_thd('min_split_score')

        use_conf_score = self.str_exist('use_conf_score')
        pad_fg_pixel = self.extract_thd('pad_fg_pixel')
        # proposal_token_scoring_gt_criterion = self.extract_sub_setting('proposal_token_scoring_gt_criterion')

        return dict(min_fg_score=min_fg_score,
                    min_split_score=min_split_score,
                    use_conf_score=use_conf_score,
                    pad_fg_pixel=pad_fg_pixel,
                    # token_scoring_gt_criterion=proposal_token_scoring_gt_criterion
                    )


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

        # delete area information to avoid possible mis-using
        del targets_proposal[k]['original_area']
        del targets_proposal[k]['area']

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