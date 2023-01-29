import cv2
import os
import smrc.utils
import numpy as np

from TCFormer.tcformer_module.tcformer_utils import *
import math

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
MIN_FG_SIGNIFICANCE = 0.6
MAX_BG_SIGNIFICANCE = 0.3  # not used in this file, but in sgdt_module.


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


def prepare_sgdt_targets(targets, pad_fg_pixel=16, token_scoring_gt_criterion=None):
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
            for kk, (box, box_area) in enumerate(zip(box_unnormalized, img_target['area'])):
                x1, y1, x2, y2 = box
                # fg_gt[k, y1:y2, x1:x2] = True  # foreground objects

                # soft label for small object so that smaller objects has large value (e.g., 1 vs. 0.9)
                #  than relative large small objects
                if token_scoring_gt_criterion.find('significance_value') > -1:
                    # significance_value_bg_w_priority

                    # significant_score = estimate_significant_score(box_area)
                    significant_score = estimate_sig_score_piecewise_linear(box_area)

                    # Use the max significant_score if overlap exists
                    scale_gt[k, y1:y2, x1:x2] = torch.max(
                        scale_gt[k, y1:y2, x1:x2],
                        scale_gt.new_tensor(significant_score)
                    )
                elif token_scoring_gt_criterion == 'fg_scale_class':
                    fg_gt[k, y1:y2, x1:x2] = True  # foreground objects

                    # TODO: soft label for small object so that smaller objects has large value (e.g., 1 vs. 0.9)
                    #  than relative large small objects
                    if is_small_object(box_area):  # small object or not
                        scale_gt[k, y1:y2, x1:x2] = True
                else:
                    raise NotImplementedError
                # else:
                #     print(f' False, {int(box_area)}  > {areaRngSGDT[2]}')

    if token_scoring_gt_criterion.find('significance_value') > -1:
        fg_gt = torch.where(scale_gt.float() > 0, 1.0, 0.0)
    # else:
    #     raise NotImplementedError

    sgdt_targets_raw = dict(
        fg_gt=fg_gt.float(),  # B, H, W
        scale_gt=scale_gt.float()  # B, H, W
    )

    return sgdt_targets_raw


# def interpolate_for_max_pool()

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


class TokenScoringGTGenerator:
    def __init__(self,
                 token_scoring_gt_criterion,
                 pad_fg_pixel=0
                 ):
        self.token_scoring_gt_criterion = token_scoring_gt_criterion

        self.pad_fg_pixel = pad_fg_pixel

        self.sig_value_interpolate_mode = 'bilinear'  # nearest cause round off error
        # if token_scoring_gt_criterion == 'significance_value_bg_w_priority':
        #     # https://imagingsolution.blog.fc2.com/blog-entry-142.html
        #     self.sig_value_interpolate_mode = 'bilinear'

    def get_gt_raw(self, targets):
        return prepare_sgdt_targets(
            targets=targets, pad_fg_pixel=self.pad_fg_pixel,
            token_scoring_gt_criterion=self.token_scoring_gt_criterion)

    def resize_sig_value_gt(self, sgdt_targets, feat_map_size):
        return resize_sgdt_target(
            sgdt_targets=sgdt_targets,
            feat_map_size=feat_map_size,
            interpolate_mode=self.sig_value_interpolate_mode
        )


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

    def __init__(self, targets, sgdt_target_raw, sgdt_targets, sgdt_output):
        self.targets = targets
        self.sgdt_target_raw = sgdt_target_raw
        self.sgdt_output = sgdt_output
        self.sgdt_targets=sgdt_targets

        self.out_dir = '/disks/cnn1/kaikai/project/DN-DETR/visualize'

    def visualize_token_adaption(self):
        # self.visualize_fg_scale_gt()

        # self.visualize_prediction()
        # self.visualize_l1_loss_prediction()
        self.visualize_fg_prediction()

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

    @staticmethod
    def resize_token_label_to_input_img(token_labels, feat_map_size, input_img_size):
        """
            pred_labels = pred_labels.view(-1, h, w)[:, None, :, :].repeat(
                1, 3, 1, 1)  # B, N
            pred_labels = F.interpolate(pred_labels.float(), [H, W], mode='nearest')

        Args:
            token_labels:
            feat_map_size:
            input_img_size:

        Returns:

        """
        h, w = feat_map_size
        H, W = input_img_size
        labels = token_labels.permute(1, 0).view(-1, h, w)[:, None, :, :].repeat(
            1, 3, 1, 1)  # N,B -> B, N -> B, h, w -> B, 1, h, w -> B, 3, h, w
        # labels = F.interpolate(labels.float(), [H, W], mode='bilinear').bool().float()  #
        labels = F.interpolate(labels.float(), [H, W], mode='nearest').bool().float()  #
        return labels

    def resize_token_multi_label_to_input_img(self, token_labels, feat_map_size, input_img_size):
        """
            pred_labels = pred_labels.view(-1, h, w)[:, None, :, :].repeat(
                1, 3, 1, 1)  # B, N
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
            label = self.resize_token_label_to_input_img(token_labels[:, :, k], feat_map_size, input_img_size)
            label_list.append(label)
        final_labels = torch.stack(label_list, dim=-1)

        return final_labels

    @staticmethod
    def token_label_feat_map_to_img(single_img_labels, color=smrc.utils.WHITE):
        img = single_img_labels.permute(1, 2, 0).detach().cpu().float().numpy()
        final_img = np.float32(img * np.array(color))
        return final_img

    # def token_pred_feat_map_to_img(single_img_labels, color=smrc.utils.WHITE):
    #
    #
    #     img = single_img_labels.permute(1, 2, 0).detach().cpu().float().numpy()
    #     final_img = np.float32(img * np.array(color))
    #     return final_img



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

    @staticmethod
    def token_mask_to_color_img(pred_mask):
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
            gt_feat_map = F.interpolate(gt_feat_map.float(), [H, W], mode='bilinear').bool().float()   # torch.Size([2, 3, 1105, 736])

            pred_labels = self.resize_token_label_to_input_img(pred_labels, feat_map_size=(h, w), input_img_size=(H, W))

            # -----------------------------
            pred_mask = self.resize_token_multi_label_to_input_img(
                pred_mask, feat_map_size=(h, w), input_img_size=(H, W))
            pred_sampled_token_label = self.resize_token_label_to_input_img(
                pred_sampled_token_label, feat_map_size=(h, w), input_img_size=(H, W))
            pred_fixed_num_token_label = self.resize_token_label_to_input_img(
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
                feat_img = self.token_label_feat_map_to_img(gt_feat_map[k])
                # img_path = os.path.join(out_dir, f'{img_id}_gt_{gt_name}.jpg')
                # cv2.imwrite(img_path, feat_img)
                final_img = cv2.addWeighted(ori_img, 0.5, feat_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_gt_{gt_name}_overlap.jpg')
                cv2.imwrite(img_path, final_img)

                # pred_img = pred_labels[k].permute(1, 2, 0).detach().cpu().float().numpy() * 255
                pred_img = self.token_label_feat_map_to_img(pred_labels[k])
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}.jpg')
                cv2.imwrite(img_path, pred_img)

                final_img = cv2.addWeighted(ori_img, 0.5, pred_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_overlap.jpg')
                cv2.imwrite(img_path, final_img)

                #
                # pred_img = pred_mask[k].permute(1, 2, 0).cpu().float().numpy() # * 255
                # pred_img = pred_img * np.array(smrc.utils.color.RED).reshape(3, 1, 1) + \
                #            (1 - pred_img) * np.array(smrc.utils.color.YELLOW).reshape(3, 1, 1)
                pred_img = self.token_mask_to_color_img(pred_mask[k])
                # fg red, bg yellow.
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_mask.jpg')
                cv2.imwrite(img_path, pred_img)
                final_img = cv2.addWeighted(ori_img, 0.5, pred_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_mask_overlap.jpg')
                cv2.imwrite(img_path, final_img)
                # --------------
                pred_img = self.token_label_feat_map_to_img(pred_sampled_token_label[k])
                # self.token_mask_to_color_img(pred_sampled_token_label[k])
                # fg red, bg yellow.
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_pred_sampled_token_label.jpg')
                cv2.imwrite(img_path, pred_img)
                final_img = cv2.addWeighted(ori_img, 0.5, pred_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_pred_sampled_token_label_overlap.jpg')
                cv2.imwrite(img_path, final_img)

                pred_img = self.token_label_feat_map_to_img(pred_fixed_num_token_label[k])
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
            pred_significance_score = F.interpolate(pred_significance_score.float(), [H, W], mode='nearest')  # torch.Size([2, 3, 1105, 736])

            # pred_labels = self.resize_token_label_to_input_img(pred_labels, feat_map_size=(h, w), input_img_size=(H, W))
            # # -----------------------------
            # pred_mask = self.resize_token_multi_label_to_input_img(
            #     pred_mask, feat_map_size=(h, w), input_img_size=(H, W))

            pred_sampled_token_label = self.resize_token_label_to_input_img(
                pred_sampled_token_label, feat_map_size=(h, w), input_img_size=(H, W))
            pred_final_token_label = self.resize_token_label_to_input_img(
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
                feat_img = self.token_label_feat_map_to_img(gt_feat_map[k])
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

    def visualize_fg_prediction(self):
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
        h, w = list(self.sgdt_output['feat_map_size'].detach().cpu().numpy())

        # new_img_size (H_new, W_new)
        img, new_img_size = self.make_img_size_divisible(img, feat_map_size=(h, w))

        B, C, H, W = img.shape  # (Tensor[B, 3, H, W])
        fg_gt_raw, scale_gt_raw = self.sgdt_target_raw['fg_gt'], self.sgdt_target_raw['scale_gt']
        # update the gt_raw if the original imgs are updated
        fg_gt_raw = self.paded_imgs(fg_gt_raw, new_img_size=new_img_size)
        scale_gt_raw = self.paded_imgs(scale_gt_raw, new_img_size=new_img_size)
        # -------------------- gt of feature map size

        # sgdt_targets = resize_sgdt_target(self.sgdt_target_raw,
        #                                   feat_map_size=self.sgdt_output['feat_map_size'],
        #                                   interpolate_mode='bilinear'
        #                                   )
        fg_gt, scale_gt = self.sgdt_targets['fg_gt'], self.sgdt_targets['scale_gt']
        N, _ = fg_gt.shape

        # predicted as to discard but actually is fg
        fg_tokens_missed = tokens_to_discard.float() * fg_gt.float()
        bg_tokens_correct = tokens_to_discard.float() * (1 - fg_gt.float())
        bg_tokens_missed = (1 - tokens_to_discard.float()) * (1 - fg_gt.float())
        # torch.Size([864, 2, 3]) N, B, 3
        fg_pred_tokens_classified = torch.stack((bg_tokens_correct, fg_tokens_missed, bg_tokens_missed), dim=-1)

        # split_tokens_missed =

        # ------------------------------------------------- plot section
        for i, (gt_name, gt, gt_on_feat_map,
                # pred_labels, #
                tokens_pred_classified, fg_missed,
                pred_sampled_token_label, pred_final_token_label) in enumerate(zip(
            ['discard', 'split'],
            [fg_gt_raw, scale_gt_raw],
            [fg_gt, scale_gt],
            # [bg_mask, scale_mask],
            [fg_pred_tokens_classified, None],
            [fg_tokens_missed, None],
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
            pred_significance_score = F.interpolate(pred_significance_score.float(), [H, W], mode='nearest')  # torch.Size([2, 3, 1105, 736])

            # pred_labels = self.resize_token_label_to_input_img(pred_labels, feat_map_size=(h, w), input_img_size=(H, W))
            # # -----------------------------
            # pred_mask = self.resize_token_multi_label_to_input_img(
            #     pred_mask, feat_map_size=(h, w), input_img_size=(H, W))
            if tokens_pred_classified is not None:
                tokens_pred_classified = self.resize_token_multi_label_to_input_img(
                    tokens_pred_classified, feat_map_size=(h, w), input_img_size=(H, W))

            pred_sampled_token_label = self.resize_token_label_to_input_img(
                pred_sampled_token_label, feat_map_size=(h, w), input_img_size=(H, W))
            pred_final_token_label = self.resize_token_label_to_input_img(
                pred_final_token_label, feat_map_size=(h, w), input_img_size=(H, W))
            if fg_missed is not None:
                fg_missed = self.resize_token_label_to_input_img(
                    fg_missed, feat_map_size=(h, w), input_img_size=(H, W))

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
                feat_img = self.token_label_feat_map_to_img(gt_feat_map[k])
                # feat_img = self.token_score_feat_map_to_img(gt_feat_map[k])
                # img_path = os.path.join(out_dir, f'{img_id}_gt_{gt_name}.jpg')
                # cv2.imwrite(img_path, feat_img)
                gt_final_img = cv2.addWeighted(box_img.copy(), 0.5, feat_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_gt_{gt_name}_overlap.jpg')
                cv2.imwrite(img_path, gt_final_img)

                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_significance_score.jpg')
                smrc.utils.plot_matrix(
                    matrix=pred_significance_score[k][0].detach().cpu().numpy(),  # (3, 704, 1072)
                    plot_name=img_path)

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

                # # pred_img = pred_labels[k].permute(1, 2, 0).detach().cpu().float().numpy() * 255
                # pred_img = self.token_label_feat_map_to_img(pred_labels[k])
                # # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}.jpg')
                # # cv2.imwrite(img_path, pred_img)
                # final_img = cv2.addWeighted(ori_img, 0.5, pred_img, 0.5, 2.2)
                # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_overlap.jpg')
                # cv2.imwrite(img_path, final_img)


                # pred_img = pred_mask[k].permute(1, 2, 0).cpu().float().numpy() # * 255
                # pred_img = pred_img * np.array(smrc.utils.color.RED).reshape(3, 1, 1) + \
                #            (1 - pred_img) * np.array(smrc.utils.color.YELLOW).reshape(3, 1, 1)
                if tokens_pred_classified is not None:
                    pred_img = self.token_mask_to_color_img(tokens_pred_classified[k])
                    # fg red, bg yellow.
                    # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_mask.jpg')
                    # cv2.imwrite(img_path, pred_img)
                    final_img = cv2.addWeighted(ori_img, 0.5, pred_img, 0.5, 2.2)
                    img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_mask_overlap.jpg')
                    cv2.imwrite(img_path, final_img)

                # --------------
                pred_img = self.token_label_feat_map_to_img(pred_sampled_token_label[k])
                # # self.token_mask_to_color_img(pred_sampled_token_label[k])
                # # fg red, bg yellow.
                # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_pred_raw_sampled_token_label.jpg')
                # cv2.imwrite(img_path, pred_img)
                final_img = cv2.addWeighted(box_img.copy(), 0.5, pred_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_raw_sampled_token_label_overlap.jpg')
                cv2.imwrite(img_path, final_img)

                pred_img = self.token_label_feat_map_to_img(pred_final_token_label[k])
                # # fg red, bg yellow.
                # img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_pred_final_token_label.jpg')
                # cv2.imwrite(img_path, pred_img)
                final_img = cv2.addWeighted(box_img.copy(), 0.5, pred_img, 0.5, 2.2)
                img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_final_token_label_overlap.jpg')
                cv2.imwrite(img_path, final_img)

                # plot missed fg tokens
                if fg_missed is not None:
                    pred_img = self.token_label_feat_map_to_img(fg_missed[k])
                    final_img = cv2.addWeighted(box_img.copy(), 0.5, pred_img, 0.5, 2.2)
                    img_path = os.path.join(out_dir, f'{img_id}_pred_{gt_name}_missed.jpg')
                    cv2.imwrite(img_path, final_img)

                # # draw predicted significant scores.


def vis_tokens_sgdt(img, token_dict, edge_color=[1.0, 1.0, 1.0], edge_width=1):
    """Visualize tokens
    Return:
        vis_img (Tensor[B, 3, H, W]): visualize result.

    Args:
        img (Tensor[B, 3, H, W]): input image.
        token_dict (dict): dict for input token information
        edge_color (float[int]): color for edges
        edge_width (int): width for edges
    """

    N = token_dict['token_num']
    device, dtype = img.device, img.dtype

    B, C, H, W = img.shape

    # ---------------
    h, w = token_dict['map_size']
    feat_map = token_dict['x'].permute(1, 0).view(-1, h, w)[:, None, :, :].repeat(1, 3, 1, 1)  # N,B -> B, 3, H, W,
    feat_map = F.interpolate(feat_map.float(), [H, W], mode='bilinear')
    # ---------------

    out_dir = '/disks/cnn1/kaikai/project/DN-DETR/'

    for k in range(B):
        img_path = os.path.join(out_dir, f'test{k}.jpg')
        ori_img = img[k].permute(1, 2, 0).cpu().numpy() * 255
        feat_img = feat_map[k].permute(1, 2, 0).cpu().numpy() * 255
        final_img = cv2.addWeighted(ori_img, 0.5, feat_img, 0.5, 2.2)
        cv2.imwrite(img_path, final_img)

        img_path = os.path.join(out_dir, f'original_img{k}.jpg')
        cv2.imwrite(img_path, ori_img)

        img_path = os.path.join(out_dir, f'feat_img{k}.jpg')
        cv2.imwrite(img_path, feat_img)
    return feat_map  # torch.Size([2, 3, 2208, 1472])

    # # color_map = torch.tensor(img, device=device, dtype=float) / 255.0
    # # color_map = color_map.permute(2, 0, 1)[None, ...]
    # color_map = F.avg_pool2d(img, kernel_size=4)  # reduce the resolution to 1/4?
    #
    # B, C, H, W = color_map.shape  # torch.Size([2, 3, 1105, 736]) -> torch.Size([2, 3, 276, 184])
    #
    # token_color = map2token(color_map, token_dict)
    # tmp_dict = token_dict.copy()
    # tmp_dict['map_size'] = [H, W]
    # tmp_dict['x'] = token_color
    # vis_img = token2map(tmp_dict)
    #
    # token_idx = torch.arange(N, device=device)[None, :, None].float() / N
    # tmp_dict['x'] = token_idx
    # idx_map = token2map(tmp_dict)  # [B, 1, H, W]
    #
    # vis_img = F.interpolate(vis_img, [H * 8, W * 8], mode='nearest')
    # idx_map = F.interpolate(idx_map, [H * 8, W * 8], mode='nearest')
    #
    # kernel = idx_map.new_zeros([4, 1, 3, 3])
    # kernel[:, :, 1, 1] = 1
    # kernel[0, :, 0, 1] = -1
    # kernel[1, :, 2, 1] = -1
    # kernel[2, :, 1, 0] = -1
    # kernel[3, :, 1, 2] = -1
    #
    # for i in range(edge_width):
    #     edge_map = F.conv2d(F.pad(idx_map, [1, 1, 1, 1], mode='replicate'), kernel)
    #     edge_map = (edge_map != 0).max(dim=1, keepdim=True)[0]
    #     idx_map = idx_map * (~edge_map) + torch.rand(idx_map.shape, device=device, dtype=dtype) * edge_map
    #
    # edge_color = torch.tensor(edge_color, device=device, dtype=dtype)[None, :, None, None]
    # vis_img = vis_img * (~edge_map) + edge_color * edge_map
    # return vis_img  # torch.Size([2, 3, 2208, 1472])


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
