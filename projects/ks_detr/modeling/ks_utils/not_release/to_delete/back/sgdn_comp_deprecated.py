import os

import cv2
import torch

import smrc.utils
from models.sgdt.sgdn_components import resize_sgdt_target, unnormalize_box


def visualize_token_adaption(targets, sgdt_target_raw, sgdt_output):
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
                        fg_score_logits=fg_score_logits,  # B, N; torch.Size([2, 630])
                        small_scale_score_logits=small_scale_score_logits,
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

    img = targets[-1]['input_imgs']
    B, C, H, W = img.shape  # (Tensor[B, 3, H, W])

    fg_gt_raw, scale_gt_raw = sgdt_target_raw['fg_gt'], sgdt_target_raw['scale_gt']
    sgdt_targets = resize_sgdt_target(sgdt_target_raw, feat_map_size=sgdt_output['feat_map_size'])
    fg_gt, scale_gt = sgdt_targets['fg_gt'], sgdt_targets['scale_gt']
    N, _ = fg_gt.shape

    # # ---------------- to remove
    # color_map = F.avg_pool2d(imgs, kernel_size=4)  # reduce the resolution to 1/4?
    # B, C, H, W = color_map.shape  # torch.Size([2, 3, 1105, 736]) -> torch.Size([2, 3, 276, 184])
    # N = H * W
    # # ----------------
    h, w = list(sgdt_output['feat_map_size'].cpu().numpy())
    device = fg_gt_raw.device
    idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)  # B, N
    # idx_token[:, :200] = 3
    agg_weight = fg_gt_raw.new_ones(B, N, 1)
    gt_token_dict = {
            'x': fg_gt_raw,
            'token_num': N,
            'map_size': [h, w],
            'init_grid_size': [h, w],  # [H, W]
            'idx_token': idx_token,
            'agg_weight': agg_weight
        }
    # N = token_dict['token_num']
    # device, dtype = img.device, img.dtype
    # B, C, H, W = img.shape

    # ---------------
    # h, w = token_dict['map_size']

    # ---------------

    out_dir = '/disks/cnn1/kaikai/project/DN-DETR/visualize'

    for id, (gt, gt_name) in enumerate(zip([fg_gt_raw, scale_gt_raw], ['fg_gt', 'scale_gt'])):
        # feat_map = gt.permute(1, 0).view(-1, h, w)[:, None, :, :].repeat(1, 3, 1, 1)  # N,B -> B, 3, H, W,
        # feat_map = F.interpolate(feat_map.float(), [H, W], mode='nearest')
        feat_map = gt[:, None, :, :].repeat(1, 3, 1, 1)  # B, H, W -> B, 3, H, W,

        for k in range(B):

            img_id = targets[k]['image_id'].cpu().item()
            ori_img = img[k].permute(1, 2, 0).cpu().numpy() * 255

            box_img = ori_img.copy()
            img_target = targets[k]
            # input_img_size = img_target['size']
            num_box = len(img_target['boxes'])
            box_unnormalized = unnormalize_box(box_normalized=img_target['boxes'],
                                               input_img_size=img_target['size'])

            if num_box > 0:  # tensor([32.0000, 31.6452], device='cuda:0')
                for box in box_unnormalized:
                    x1, y1, x2, y2 = box
                    box_img = cv2.rectangle(box_img, (int(x1), int(y1)), (int(x2), int(y2)), smrc.utils.YELLOW, 4)

            img_path = os.path.join(out_dir, f'{img_id}_box_img.jpg')
            cv2.imwrite(img_path, box_img)

            ori_img = box_img.copy()
            # feat_img = feat_map[k].permute(1, 2, 0).cpu().numpy() * 255
            feat_img = feat_map[k].permute(1, 2, 0).cpu().float().numpy() * 255
            final_img = cv2.addWeighted(ori_img, 0.5, feat_img, 0.5, 2.2)
            img_path = os.path.join(out_dir, f'{img_id}_final_img_{gt_name}_overlap.jpg')
            cv2.imwrite(img_path, final_img)

            img_path = os.path.join(out_dir, f'{img_id}_original_img.jpg')
            cv2.imwrite(img_path, ori_img)

            img_path = os.path.join(out_dir, f'{img_id}_feat_img_{gt_name}.jpg')
            cv2.imwrite(img_path, feat_img)

            # ================
            # if id == 0:


    # return feat_map  # torch.Size([2, 3, 2208, 1472])

    #
    # vis_img = vis_tokens_sgdt(img=imgs, token_dict=gt_token_dict)
    # out_dir = '/disks/cnn1/kaikai/project/DN-DETR/'

    # for k in range(vis_img.shape[0]):
    #     img_path = os.path.join(out_dir, f'test{k}.jpg')
    #     img = vis_img[k].permute(1, 2, 0).cpu().numpy() * 255
    #     cv2.imwrite(img_path, img)



# def prepare_sgdt_targets(targets, pad_fg_pixel=32):
#     """
#
#     Args:
#         targets: a list of dict, each dict contains the gt information of one image.
#         each dict:
#         'boxes' = {Tensor: 4} tensor([[0.5000, 0.5921, 1.0000, 0.8157],\n        [0.6338, 0.6836, 0.6812, 0.6058],\n        [0.5718, 0.2573, 0.4123, 0.5145],\n        [0.2712, 0.9666, 0.5423, 0.0669]], device='cuda:0')
#         'labels' = {Tensor: 4} tensor([67, 47, 47, 60], device='cuda:0')
#         'image_id' = {Tensor: 1} tensor([276037], device='cuda:0')
#         'area' = {Tensor: 4} tensor([425523.2500, 215284.8281, 110670.8125,  18919.1719], device='cuda:0')
#         'iscrowd' = {Tensor: 4} tensor([0, 0, 0, 0], device='cuda:0')
#         'orig_size' = {Tensor: 2} tensor([640, 448], device='cuda:0') # img_h, img_w = tgt['orig_size'].unbind()
#         'size' = {Tensor: 2} tensor([741, 704], device='cuda:0'), ([int(h), int(w)])
#     Returns:
#            sgdt_targets_raw:  # cannot be used as gt.
#                dict(
#                     fg_gt=fg_gt,  # B, H, W
#                     scale_gt=scale_gt  #  B, H, W
#                 )
#     Due to random crop in the transpose,
#     original box are is 1804, but in the cropped image, it occupies the whole image.
#      True, 1804  < 9216, tensor([[0.5000, 0.5000, 1.0000, 1.0000],
#         [0.5000, 0.5000, 1.0000, 1.0000]], device='cuda:0')
#     """
#     # B, H, W
#     padded_img_size = targets[0]['padded_img_size']  # (736, 981)
#     batch_size = len(targets)
#     mask_size = (batch_size,) + tuple(padded_img_size.cpu().numpy())
#
#     # We must not use scale_gt = fg_gt = torch.zeros(), otherwise, they will share the same variable.
#     fg_gt = torch.zeros(mask_size).to(targets[0]['size'].device).bool()  # H, W
#     scale_gt = torch.zeros(mask_size).to(targets[0]['size'].device).bool()
#
#     for k, img_target in enumerate(targets):
#         # 0 means bg, 1, fg. -1 means padding position.
#         box_unnormalized = unnormalize_box(box_normalized=img_target['boxes'],
#                                            input_img_size=img_target['size'])
#
#         num_box = len(img_target['boxes'])
#         if num_box > 0:  # tensor([32.0000, 31.6452], device='cuda:0')
#             # ------------------------- Extend the fg regions
#             if pad_fg_pixel > 0:
#                 input_img_size = img_target['size']
#                 h, w = input_img_size[0].item(), input_img_size[1].item()
#                 offset = torch.tensor([-pad_fg_pixel, -pad_fg_pixel, pad_fg_pixel, pad_fg_pixel],
#                                       dtype=torch.int32, device=box_unnormalized.device
#                                       ).unsqueeze(dim=0).repeat(num_box, 1)
#                 box_unnormalized += offset
#                 box_unnormalized[:, 0::2].clamp_(min=0, max=w)  # w: x
#                 box_unnormalized[:, 1::2].clamp_(min=0, max=h)  # h: y
#
#             # -------------------------------- Generate the gt mask
#             # Using the area of the box in the original img, instead of the input image, which will be changed in
#             # self._transforms is a good choice.
#             # But original area has more box then the final box, and we do not know the correspondence for the box list
#             # with different number of boxes, so we cannot use the original area.
#             for kk, (box, box_area) in enumerate(zip(box_unnormalized, img_target['area'])):  # 'original_area', 'area'
#                 x1, y1, x2, y2 = box
#
#                 # ================================
#                 #  v0
#                 # ================================
#                 fg_gt[k, y1:y2, x1:x2] = True  # foreground objects
#
#                 # TODO: soft label for small object so that smaller objects has large value (e.g., 1 vs. 0.9)
#                 #  than relative large small objects
#                 if is_small_object(box_area):  # small object or not
#                     # if not is_small_object(img_target['area'][kk]):
#                     #     print(f' True, {int(box_area)}  < {areaRngSGDT[2]}, {img_target["boxes"]}')
#                     scale_gt[k, y1:y2, x1:x2] = True
#                     # small_obj_mask = torch.full_like(fg_gt[k, y1:y2, x1:x2],
#                     #                                  is_small_object(box_area))
#                     # scale_gt[k, y1:y2, x1:x2] = torch.logical_or(
#                     #     scale_gt[k, y1:y2, x1:x2], small_obj_mask
#                     # )  # as long as the box region covers a small object, the label is set True, no matter whether
#                 # # a large object also appears in the same region.
#                 # else:
#                 #     print(f' False, {int(box_area)}  > {areaRngSGDT[2]}')
#     sgdt_targets_raw = dict(
#         fg_gt=fg_gt,  # B, H, W
#         scale_gt=scale_gt  # B, H, W
#     )
#
#     return sgdt_targets_raw

# def resize_sgdt_target(sgdt_targets, feat_map_size, feat_map_mask=None):
#     """
#
#     Args:
#         sgdt_targets:
#         feat_map_size:
#         feat_map_mask: (B, H, W), bool, True means padded tokens (invalid, not be used in computation)
#
#     Returns:
#
#     """
#     fg_gt, scale_gt = sgdt_targets['fg_gt'], sgdt_targets['scale_gt']
#     # B, H, W -> B, h, w (feature map size) size=x.shape[-2:]
#     # feat_map_size = sgdt_output['feat_map_size']
#     if torch.is_tensor(feat_map_size):
#         output_size = tuple(feat_map_size.cpu().numpy())
#     else:
#         # if not isinstance(feat_map_size, (tuple, list)):
#         output_size = tuple(feat_map_size)
#
#     fg_gt = F.interpolate(fg_gt[None].float(), size=output_size).to(torch.bool)[0]  # torch.Size([2, 23, 31])
#     scale_gt = F.interpolate(scale_gt[None].float(), size=output_size).to(torch.bool)[0]
#
#     # ======================== only for debugging, TODO: remove the following lines
#     # no need to do the following operation
#     if feat_map_mask is not None:
#         ErrorFlag = False
#         if fg_gt[feat_map_mask].sum() > 0:
#             print(f'fg_gt[feat_map_mask].sum() = {fg_gt[feat_map_mask].sum()}')
#             ErrorFlag = True
#         if scale_gt[feat_map_mask].sum() > 0:
#             print(f'fg_gt[feat_map_mask].sum() = {scale_gt[feat_map_mask].sum()}')
#             ErrorFlag = True
#         if ErrorFlag:
#             raise ErrorFlag
#
#         fg_gt[feat_map_mask] = False
#         scale_gt[feat_map_mask] = False
#     # ========================
#
#     sgdt_targets = dict(  # B, H, W -> HxW, B, and then will be expanded to (N, B, Num_Class)
#         fg_gt=fg_gt.flatten(1).permute(1, 0).long(),
#         scale_gt=scale_gt.flatten(1).permute(1, 0).long()
#     )
#     return sgdt_targets


class VisualizeToken_v0:
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
                            fg_score_logits=fg_score_logits,  # B, N; torch.Size([2, 630])
                            small_scale_score_logits=small_scale_score_logits,
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

    def __init__(self, targets, sgdt_target_raw, sgdt_output):
        self.targets = targets
        self.sgdt_target_raw = sgdt_target_raw
        self.sgdt_output = sgdt_output

        self.out_dir = '/disks/cnn1/kaikai/project/DN-DETR/visualize'

    def visualize_token_adaption(self):
        # self.visualize_fg_scale_gt()

        self.visualize_prediction()

    @staticmethod
    def draw_token_edge(img, input_img_size, feat_map_size):
        # ==========================
        # draw the token boundary
        H, W = input_img_size
        h, w = feat_map_size

        edge_color = tuple([1.0, 0.0, 1.0])  # [1.0, 1.0, 1.0]
        stride_h, stride_w = H / h, W / w
        for ii in range(H // h + 1):
            y = int(stride_h * ii)
            img = cv2.line(img, (0, y), (W, y), edge_color, 2)

        for ii in range(W // w + 1):
            x = int(stride_w * ii)
            img = cv2.line(img, (x, 0), (x, H), edge_color, 2)
        # ==========================
        return img

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
            gt_feat_map = F.interpolate(gt_feat_map.float(), [H, W], mode='nearest')

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

        img_resized = F.interpolate(img, [h, w], mode='nearest')
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
        labels = F.interpolate(labels.float(), [H, W], mode='nearest')
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
            label = self.resize_token_label_to_input_img(token_labels[:, :, k], feat_map_size, input_img_size)
            label_list.append(label)
        final_labels = torch.stack(label_list, dim=-1)

        return final_labels

    @staticmethod
    def token_label_feat_map_to_img(single_img_labels):
        final_img = single_img_labels.permute(1, 2, 0).detach().cpu().float().numpy() * 255
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
            np.array(smrc.utils.color.BLACK),
            np.array(smrc.utils.color.RED),
            np.array(smrc.utils.color.Pink),
        ]
        # 3, H, W, Num_label -> H, W, 3, Num_label (1105, 736, 3, 2)
        pred_img = pred_mask.detach().cpu().permute(1, 2, 0, 3).float().numpy()  # * 255
        mask = np.zeros_like(pred_img[:, :, :, 0])  # H, W, 3,

        for k in range(num_label):
            class_mask = pred_img[:, :, :, k] == 1  # (1105, 736, 3)

            mask += class_mask * colors[k].reshape(1, 1, 3)
        return mask

    def visualize_prediction(self):
        """
            # ---------------- for visualization purpose
            fg_mask=adapted_token_dict['fg_mask'],
            fg_obj_score=adapted_token_dict['fg_obj_score'],
            tokens_to_discard_original=adapted_token_dict['tokens_to_discard_original'],
            small_scale_score=adapted_token_dict['small_scale_score'],
            scale_mask=adapted_token_dict['scale_mask'],
            tokens_small_obj_original=adapted_token_dict['tokens_small_obj_original'],
        Returns:

        """

        # adapted_pos = self.sgdt_output['adapted_pos']

        # fg_score = self.sgdt_output['fg_obj_score']  # B, N; torch.Size([2, 630])
        # small_scale_score = self.sgdt_output['small_scale_score']

        fg_score_logits = self.sgdt_output['fg_score_logits']  # N, B, Num_class
        small_scale_score_logits = self.sgdt_output['small_scale_score_logits']
        fg_labels = fg_score_logits.max(dim=-1).indices
        small_scale_labels = small_scale_score_logits.max(dim=-1).indices

        fg_mask = self.sgdt_output['fg_mask']  # N, B, Num_class
        scale_mask = self.sgdt_output['scale_mask']  # N, B, Num_class

        tokens_to_discard_original = self.sgdt_output['tokens_to_discard_original']
        tokens_small_obj_original = self.sgdt_output['tokens_small_obj_original']

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
            [tokens_to_discard_original, tokens_small_obj_original],
            [tokens_to_discard, tokens_small_obj],
        )):
            gt_raw = gt[:, None, :, :].repeat(1, 3, 1, 1)  # B, H, W -> B, 3, H, W,

            gt_feat_map = gt_on_feat_map.permute(1, 0).view(-1, h, w)[:, None, :, :].repeat(
                1, 3, 1, 1)  # N,B -> B, 3, H, W,
            gt_feat_map = F.interpolate(gt_feat_map.float(), [H, W], mode='nearest')  # torch.Size([2, 3, 1105, 736])

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


# class TokenScoringTarget:
#     def __init__(self, max_box_area=1024 ** 2):
#         self.max_box_area = max_box_area


MAX_BOX_AREA = 1024 ** 2


def estimate_significant_score(self, box_area):
    # smaller area has larger score

    # clip the box_area if it is larger than the defined max_box_area
    box_area = min(self.max_box_area, box_area)
    significant_score = (self.max_box_area - box_area) / self.max_box_area
    # shift the score to the range of [0.5, 1], so that fg tokens has value >= 0.5
    significant_score = 0.5 + significant_score / 2.0
    return significant_score


def prepare_sgdt_targets(self, targets, pad_fg_pixel=32):
    """

    Args:
        pad_fg_pixel:
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
    # B, H, W
    padded_img_size = targets[0]['padded_img_size']  # (736, 981)
    batch_size = len(targets)
    mask_size = (batch_size,) + tuple(padded_img_size.cpu().numpy())

    # We must not use scale_gt = fg_gt = torch.zeros(), otherwise, they will share the same variable.
    # fg_gt = torch.zeros(mask_size).to(targets[0]['size'].device).float()  # H, W  TODO
    scale_gt = torch.zeros(mask_size).to(targets[0]['size'].device).float()

    padded_img_area = torch.prod(padded_img_size)
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
                significant_score = self.estimate_significant_score(box_area)

                # Use the max significant_score if overlap exists
                scale_gt[k, y1:y2, x1:x2] = torch.max(
                    scale_gt[k, y1:y2, x1:x2],
                    scale_gt.new_tensor(significant_score)
                )

                # else:
                #     print(f' False, {int(box_area)}  > {areaRngSGDT[2]}')
    # fg_gt = scale_gt > 0
    sgdt_targets_raw = dict(
        # fg_gt=fg_gt,  # B, H, W
        scale_gt=scale_gt  # B, H, W
    )

    return sgdt_targets_raw


def resize_sgdt_target(sgdt_targets, feat_map_size, feat_map_mask=None):
    """

    Args:
        sgdt_targets:
        feat_map_size:
        feat_map_mask: (B, H, W), bool, True means padded tokens (invalid, not be used in computation)

    Returns:

    """
    # fg_gt, scale_gt = sgdt_targets['fg_gt'], sgdt_targets['scale_gt']
    scale_gt = sgdt_targets['scale_gt']
    # B, H, W -> B, h, w (feature map size) size=x.shape[-2:]
    # feat_map_size = sgdt_output['feat_map_size']
    if torch.is_tensor(feat_map_size):
        output_size = tuple(feat_map_size.cpu().numpy())
    else:
        # if not isinstance(feat_map_size, (tuple, list)):
        output_size = tuple(feat_map_size)

    # fg_gt = F.interpolate(fg_gt[None].float(), size=output_size).to(torch.bool)[0]  # torch.Size([2, 23, 31])
    scale_gt = F.interpolate(scale_gt[None].float(),
                             size=output_size, mode='nearest', ).to(torch.bool)[0]

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
        # fg_gt=fg_gt.flatten(1).permute(1, 0).long(),
        scale_gt=scale_gt.flatten(1).permute(1, 0).long()
    )
    return sgdt_targets
