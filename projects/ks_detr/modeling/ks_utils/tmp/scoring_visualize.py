import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn import functional as F

import smrc.utils
from models.ksgt.scoring_gt import unnormalize_box, resize_ksgt_target, INTERPOLATE_MODE
from models.ksgt.scoring_eval import get_pred_fg_tokens_classified, get_pred_split_tokens_classified
from tti.tti_conf import VISUALIZATION_DIR
from datetime import datetime


VISUALIZATION_INTERPOLATE_MODE = 'nearest'  # 'bilinear' nearest
# assert VISUALIZATION_INTERPOLATE_MODE == INTERPOLATE_MODE


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
            ksgt_target_raw: a dict,
                        fg_gt=fg_gt,  # B, H, W
                        scale_gt=scale_gt  #  B, H, W

            # ksgt_targets = dict(  # B, H, W -> HxW, B, and then will be expanded to (N, B, Num_Class)
            #         fg_gt=fg_gt.flatten(1).permute(1, 0), # N, B
            #         scale_gt=scale_gt.flatten(1).permute(1, 0)   #  N, B
            # )

            ksgt_output:  a dict,
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

    def __init__(self, targets, ksgt_target_raw, ksgt_targets, ksgt_output, out_dir=None):
        self.targets = targets
        self.ksgt_target_raw = ksgt_target_raw
        self.ksgt_output = ksgt_output
        self.ksgt_targets = ksgt_targets

        if out_dir is None:
            out_dir = VISUALIZATION_DIR
        self.out_dir = out_dir

    def visualize_token_adaption(self, sub_dir=None):

        # self.visualize_fg_scale_gt()

        # self.visualize_prediction()
        # self.visualize_l1_loss_prediction()
        self.visualize_split_token(
            sub_dir=f'fg_pred_view{self.datetime_str()}' if sub_dir is None else sub_dir
        )
        # self.visualize_fg_scale_prediction(
        #     light_view=True,
        #     only_padded_image=True,
        #     disable_remove_view=True,
        #     sub_dir=f'fg_split_view{self.datetime_str()}' if sub_dir is None else sub_dir
        # )  #
        # self.visualize_fg_scale_prediction(
        #     light_view=True,
        #     only_padded_image=True,  # False
        #     disable_remove_view=False,
        #     sub_dir=f'fg_ambiguous{self.datetime_str()}' if sub_dir is None else sub_dir
        # )  # fg_split_view
        # self.visualize_fg_scale_prediction(
        #     light_view=True, only_padded_image=True, disable_remove_view=True, sub_dir=sub_dir)  #

    @staticmethod
    def datetime_str():
        return datetime.now().strftime("%Y%m%d-%H")

    def visualize_split_token(self, sub_dir=None):
        # self.visualize_fg_scale_prediction(
        #     light_view=True,
        #     only_padded_image=True,
        #     disable_remove_view=True,
        #     sub_dir=f'fg_split_view{self.datetime_str()}' if sub_dir is None else sub_dir
        # )  #

        self.visualize_split_prediction(
            light_view=True,
            only_padded_image=False,
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
        # labels = F.interpolate(labels.float(), [H, W], mode=VISUALIZATION_INTERPOLATE_MODE).bool().float()  #
        labels = F.interpolate(labels.float(), [H, W], mode=VISUALIZATION_INTERPOLATE_MODE).bool().float()  #
        return labels

    def resize_token_one_hot_label_to_input_img(self, token_labels, feat_map_size, input_img_size):
        """
            pred_labels = pred_labels.view(-1, h, w)[:, None, :, :].repeat(
                1, 3, 1, 1)  #
            pred_labels = F.interpolate(pred_labels.float(), [H, W], mode=VISUALIZATION_INTERPOLATE_MODE)

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
    def _draw_significace_score(gt_sig_value_matrix, pred_sig_value_matrix, plot_name, thds=None):
        # define a list of markevery cases to plot
        if thds is None:
            thds = np.linspace(0, 0.9, 10)
        max_thd_num = len(thds) - 1

        # data points
        fig, axs = plt.subplots(3, 3, figsize=(10, 6))  # , constrained_layout=True
        for k, (ax, thd) in enumerate(zip(axs.flat, thds[:min(max_thd_num, 9)])):
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
            self.ksgt_target_raw,
            self.ksgt_output,
            self.ksgt_targets,
        ])

    def save_significant_score_and_split_tokens(self, sub_dir=None):
        sub_dir = f'fg_significant_score_and_split_tokens{self.datetime_str()}' if sub_dir is None else sub_dir

        result_file_path = ''
        for target in self.targets:
            img_id = target['image_id'].detach().cpu().item()
            result_file_path += f'_{img_id}'

        significance_score = self.ksgt_output['small_scale_score']  # N, B
        tokens_to_discard_original = self.ksgt_output['tokens_to_discard_original'].long()
        tokens_to_split_original = self.ksgt_output['tokens_to_split_original'].long()
        tokens_small_obj = self.ksgt_output['tokens_small_obj']
        tokens_to_discard = self.ksgt_output['tokens_to_discard']
        fg_gt, scale_gt = self.ksgt_targets['fg_gt'], self.ksgt_targets['scale_gt']

        result_path = os.path.join(self.out_dir, sub_dir + '_split_pkl', f'{result_file_path}.pkl')
        smrc.utils.generate_dir_for_file_if_not_exist(result_path)
        smrc.utils.generate_pkl_file(pkl_file_name=result_path, data=[
            significance_score,
            tokens_small_obj,
            scale_gt,
            # self.targets,
            # self.ksgt_target_raw,
            # self.ksgt_output,
            # self.ksgt_targets,
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

        significance_score = self.ksgt_output['small_scale_score']  # N, B
        tokens_to_discard_original = self.ksgt_output['tokens_to_discard_original'].long()
        tokens_to_split_original = self.ksgt_output['tokens_to_split_original'].long()
        # bg_mask, scale_mask = F.one_hot(tokens_to_discard_original, num_classes=2).float(), \
        #                       F.one_hot(tokens_to_split_original, num_classes=2).float()

        tokens_small_obj = self.ksgt_output['tokens_small_obj']
        tokens_to_discard = self.ksgt_output['tokens_to_discard']

        img = self.targets[-1]['input_imgs']
        h, w = list(self.ksgt_output['feat_map_size'].detach().cpu().numpy())
        valid_tokens = self.ksgt_output['valid_tokens'].bool()  # N, B
        # new_img_size (H_new, W_new)
        img, new_img_size = self.make_img_size_divisible(img, feat_map_size=(h, w))

        B, C, H, W = img.shape  # (Tensor[B, 3, H, W])
        # -------------------- raw gt
        fg_gt_raw, scale_gt_raw = self.ksgt_target_raw['fg_gt'], self.ksgt_target_raw['scale_gt']
        # update the gt_raw if the original imgs are updated
        fg_gt_raw = self.paded_imgs(fg_gt_raw, new_img_size=new_img_size)
        scale_gt_raw = self.paded_imgs(scale_gt_raw, new_img_size=new_img_size)
        fg_gt, scale_gt = self.ksgt_targets['fg_gt'], self.ksgt_targets['scale_gt']

        # -----------------------------------------------
        fg_pred_tokens_classified = get_pred_fg_tokens_classified(
            tokens_to_discard=tokens_to_discard, fg_gt=fg_gt)
        fg_original_pred_tokens_classified = get_pred_fg_tokens_classified(
            tokens_to_discard=tokens_to_discard_original, fg_gt=fg_gt)
        split_pred_tokens_classified = get_pred_split_tokens_classified(
            tokens_to_split=tokens_small_obj, split_gt=scale_gt)
        split_original_pred_tokens_classified = get_pred_split_tokens_classified(
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
                                                    mode=VISUALIZATION_INTERPOLATE_MODE)  # torch.Size([2, 3, 1105, 736])

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

        significance_score = self.ksgt_output['small_scale_score']  # N, B
        tokens_to_discard_original = self.ksgt_output['tokens_to_discard_original'].long()
        tokens_to_split_original = self.ksgt_output['tokens_to_split_original'].long()
        # bg_mask, scale_mask = F.one_hot(tokens_to_discard_original, num_classes=2).float(), \
        #                       F.one_hot(tokens_to_split_original, num_classes=2).float()

        tokens_small_obj = self.ksgt_output['tokens_small_obj']
        tokens_to_discard = self.ksgt_output['tokens_to_discard']

        img = self.targets[-1]['input_imgs']
        feat_map_size = self.ksgt_output['feat_map_size']
        if torch.is_tensor(feat_map_size):
            feat_map_size = self.ksgt_output['feat_map_size'].detach().cpu().numpy()
        h, w = list(feat_map_size)

        valid_tokens = self.ksgt_output['valid_tokens'].bool()  # N, B
        # new_img_size (H_new, W_new)
        img, new_img_size = self.make_img_size_divisible(img, feat_map_size=(h, w))

        B, C, H, W = img.shape  # (Tensor[B, 3, H, W])
        # -------------------- raw gt
        fg_gt_raw, scale_gt_raw = self.ksgt_target_raw['fg_gt'], self.ksgt_target_raw['scale_gt']
        # update the gt_raw if the original imgs are updated
        fg_gt_raw = self.paded_imgs(fg_gt_raw, new_img_size=new_img_size)
        scale_gt_raw = self.paded_imgs(scale_gt_raw, new_img_size=new_img_size)
        fg_gt, scale_gt = self.ksgt_targets['fg_gt'], self.ksgt_targets['scale_gt']

        # -----------------------------------------------
        fg_pred_tokens_classified = get_pred_fg_tokens_classified(
            tokens_to_discard=tokens_to_discard, fg_gt=fg_gt)
        fg_original_pred_tokens_classified = get_pred_fg_tokens_classified(
            tokens_to_discard=tokens_to_discard_original, fg_gt=fg_gt)
        split_pred_tokens_classified = get_pred_split_tokens_classified(
            tokens_to_split=tokens_small_obj, split_gt=scale_gt)
        split_original_pred_tokens_classified = get_pred_split_tokens_classified(
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
                                                    mode=VISUALIZATION_INTERPOLATE_MODE)  # torch.Size([2, 3, 1105, 736])

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

                # # if not light_view:
                self._draw_significace_score(
                    gt_sig_value_matrix=gt[k].detach().cpu().numpy(),
                    pred_sig_value_matrix=pred_significance_score[k][0].detach().cpu().numpy(),
                    plot_name=os.path.join(out_dir, f'{img_id}_pred_{gt_name}_significance_score_vs_gt.jpg'),
                    thds=np.linspace(0.1, 0.5, 9),
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

        significance_score = self.ksgt_output['small_scale_score']  # N, B
        tokens_to_discard_original = self.ksgt_output['tokens_to_discard_original'].long()
        tokens_to_split_original = self.ksgt_output['tokens_to_split_original'].long()
        bg_mask, scale_mask = F.one_hot(tokens_to_discard_original, num_classes=2).float(), \
                              F.one_hot(tokens_to_split_original, num_classes=2).float()

        tokens_small_obj = self.ksgt_output['tokens_small_obj']
        tokens_to_discard = self.ksgt_output['tokens_to_discard']

        # -------------------- raw gt
        img = self.targets[-1]['input_imgs']
        B, C, H, W = img.shape  # (Tensor[B, 3, H, W])
        fg_gt_raw, scale_gt_raw = self.ksgt_target_raw['fg_gt'], self.ksgt_target_raw['scale_gt']

        # -------------------- gt of feature map size
        h, w = list(self.ksgt_output['feat_map_size'].detach().cpu().numpy())
        # ksgt_targets = resize_ksgt_target(self.ksgt_target_raw,
        #                                   feat_map_size=self.ksgt_output['feat_map_size'],
        #                                   interpolate_mode=VISUALIZATION_INTERPOLATE_MODE
        #                                   )
        fg_gt, scale_gt = self.ksgt_targets['fg_gt'], self.ksgt_targets['scale_gt']
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
            # We should use the VISUALIZATION_INTERPOLATE_MODE mode to increase the resoltion, otherwise, by VISUALIZATION_INTERPOLATE_MODE, we will cause more
            # non-zero locations.
            gt_feat_map = F.interpolate(gt_feat_map.float(), [H, W],
                                        mode=VISUALIZATION_INTERPOLATE_MODE)  # torch.Size([2, 3, 1105, 736])
            # gt_feat_map = F.interpolate(gt_feat_map.float(), [H, W], mode=VISUALIZATION_INTERPOLATE_MODE).bool().float()  # torch.Size([2, 3, 1105, 736])

            pred_significance_score = significance_score.permute(1, 0).view(-1, h, w)[:, None, :, :].repeat(
                1, 3, 1, 1)  # N,B -> B, 3, H, W,
            pred_significance_score = F.interpolate(pred_significance_score.float(), [H, W],
                                                    mode=VISUALIZATION_INTERPOLATE_MODE)  # torch.Size([2, 3, 1105, 736])

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
        fg_gt_raw, scale_gt_raw = self.ksgt_target_raw['fg_gt'], self.ksgt_target_raw['scale_gt']

        # -------------------- plot on image of feature map size
        h, w = list(self.ksgt_output['feat_map_size'].cpu().numpy())
        ksgt_targets = resize_ksgt_target(self.ksgt_target_raw, feat_map_size=self.ksgt_output['feat_map_size'])
        fg_gt, scale_gt = ksgt_targets['fg_gt'], ksgt_targets['scale_gt']
        N, _ = fg_gt.shape
        # -------------------------------------------------

        for id, (gt, gt_name, gt_on_feat_map) in enumerate(zip([fg_gt_raw, scale_gt_raw],
                                                               ['fg_gt', 'scale_gt'], [fg_gt, scale_gt])):
            gt_raw = gt[:, None, :, :].repeat(1, 3, 1, 1)  # B, H, W -> B, 3, H, W,

            gt_feat_map = gt_on_feat_map.permute(1, 0).view(-1, h, w)[:, None, :, :].repeat(1, 3, 1,
                                                                                            1)  # N,B -> B, 3, H, W,
            gt_feat_map = F.interpolate(gt_feat_map.float(), [H, W], mode=VISUALIZATION_INTERPOLATE_MODE).bool().float()

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
        # h, w = list(self.ksgt_output['feat_map_size'].cpu().numpy())
        # ksgt_targets = resize_ksgt_target(self.ksgt_target_raw, feat_map_size=self.ksgt_output['feat_map_size'])
        # fg_gt, scale_gt = ksgt_targets['fg_gt'], ksgt_targets['scale_gt']
        # N, _ = fg_gt.shape

        img_resized = F.interpolate(img, [h, w], mode=VISUALIZATION_INTERPOLATE_MODE).bool().float()
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

        # adapted_pos = self.ksgt_output['adapted_pos']

        # fg_score = self.ksgt_output['fg_obj_score']  # B, N; torch.Size([2, 630])
        # small_scale_score = self.ksgt_output['small_scale_score']

        fg_score_logit = self.ksgt_output['fg_score_logit']  # N, B, Num_class
        small_scale_score_logit = self.ksgt_output['small_scale_score_logit']
        fg_labels = fg_score_logit.max(dim=-1).indices
        small_scale_labels = small_scale_score_logit.max(dim=-1).indices

        fg_mask = self.ksgt_output['fg_mask']  # N, B, Num_class
        scale_mask = self.ksgt_output['scale_mask']  # N, B, Num_class

        tokens_to_discard_original = self.ksgt_output['tokens_to_discard_original']
        tokens_to_split_original = self.ksgt_output['tokens_to_split_original']

        tokens_small_obj = self.ksgt_output['tokens_small_obj']
        tokens_to_discard = self.ksgt_output['tokens_to_discard']

        out_dir = self.out_dir

        # -------------------- plot on image of input size
        img = self.targets[-1]['input_imgs']
        B, C, H, W = img.shape  # (Tensor[B, 3, H, W])
        fg_gt_raw, scale_gt_raw = self.ksgt_target_raw['fg_gt'], self.ksgt_target_raw['scale_gt']

        # -------------------- plot on image of feature map size
        h, w = list(self.ksgt_output['feat_map_size'].detach().cpu().numpy())
        ksgt_targets = resize_ksgt_target(self.ksgt_target_raw, feat_map_size=self.ksgt_output['feat_map_size'])
        fg_gt, scale_gt = ksgt_targets['fg_gt'], ksgt_targets['scale_gt']
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
                                        mode=VISUALIZATION_INTERPOLATE_MODE).bool().float()  # torch.Size([2, 3, 1105, 736])

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
        # h, w = list(self.ksgt_output['feat_map_size'].cpu().numpy())
        # ksgt_targets = resize_ksgt_target(self.ksgt_target_raw, feat_map_size=self.ksgt_output['feat_map_size'])
        # fg_gt, scale_gt = ksgt_targets['fg_gt'], ksgt_targets['scale_gt']
        # N, _ = fg_gt.shape

        # img_resized = F.interpolate(img, [h, w], mode=VISUALIZATION_INTERPOLATE_MODE)
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


