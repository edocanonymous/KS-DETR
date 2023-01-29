import numpy as np
import torch
from tqdm import tqdm

from smrc.utils import estimate_precision_recall, estimate_F1_score, estimate_accu


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


def get_pred_fg_tokens_classified(tokens_to_discard, fg_gt):
    bg_tokens_correct, fg_tokens_missed, bg_tokens_missed = classify_predicted_remove_tokens(
        tokens_to_discard=tokens_to_discard, fg_gt=fg_gt)
    # torch.Size([864, 2, 3]) N, B, 3
    fg_pred_tokens_classified = torch.stack((bg_tokens_correct, fg_tokens_missed, bg_tokens_missed), dim=-1)
    return fg_pred_tokens_classified


def get_pred_split_tokens_classified(tokens_to_split, split_gt):
    split_tokens_correct, false_split_tokens, split_tokens_missed = classify_predicted_split_tokens(
        tokens_to_split=tokens_to_split, split_gt=split_gt)
    split_pred_tokens_classified = torch.stack(
        (split_tokens_correct, false_split_tokens, split_tokens_missed), dim=-1)
    return split_pred_tokens_classified


def eval_split_token(tokens_to_split, split_gt):
    split_tokens_correct, false_split_tokens, split_tokens_missed = classify_predicted_split_tokens(
        tokens_to_split=tokens_to_split, split_gt=split_gt)
    tp, fp, fn = split_tokens_correct.sum(), false_split_tokens.sum(), split_tokens_missed.sum()
    tn = torch.ones_like(split_tokens_correct).sum() - tp - fp - fn
    return [tp, fp, fn, tn]


def eval_remove_token(tokens_to_discard, fg_gt):
    bg_tokens_correct, fg_tokens_missed, bg_tokens_missed = classify_predicted_remove_tokens(
        tokens_to_discard=tokens_to_discard, fg_gt=fg_gt)
    tp, fp, fn = bg_tokens_correct.sum(), fg_tokens_missed.sum(), bg_tokens_missed.sum()
    tn = torch.ones_like(bg_tokens_correct).sum() - tp - fp - fn
    return [tp, fp, fn, tn]


class TokenEval:
    def __init__(self, split_or_remove='split'):

        assert split_or_remove in ['split', 'remove']
        if split_or_remove == 'split':
            self.token_classify_func = eval_split_token
        else:
            self.token_classify_func = eval_remove_token
        self._init_metric()

    def _init_metric(self):
        self.TP_ = 0
        self.FP_ = 0
        self.FN_ = 0
        self.TN_ = 0
        # self.IoUs_ = []

    def _update_metric(self, new_metric):
        """
        Update the TP, FP, FN and IoUs for the overall evaluation metrics.
        :param new_metric:
        :return:
        """
        tp, fp, fn, tn = new_metric
        self.TP_ += tp
        self.FP_ += fp
        self.FN_ += fn
        self.TN_ += tn
        # self.IoUs_ += IoUs

    def _get_cur_metric(self):
        return [self.TP_, self.FP_, self.FN_, self.TN_]

    def wrap_result(self):
        tp, fp, fn, tn = self._get_cur_metric()
        precision, recall = estimate_precision_recall(tp, fp, fn)
        accu = estimate_accu(TP=tp, FP=fp, TN=tn, FN=fn)
        F1_score = estimate_F1_score(precision, recall)
        # print(f'| tp = {tp}, fp = {fp}, fn = {fn}, tn = {tn}')
        # print(f'| precision = {precision} \n| recall = {recall} \n | accu = {accu} \n')
        # print(f'| F1_score = {F1_score}')
        return [precision, recall, F1_score, accu, tp, fp, fn, tn]

    def eval_single(
            self, token_pred, token_gt):

        singe_image_metric = self.token_classify_func(token_pred, token_gt)
        self._update_metric(singe_image_metric)


    # def _eval(
    #         self, det_file_list, gtruth_file_list):
    #     """
    #     Evaluate the precision, recall and F1 score.
    #     :param det_file_list: complete det file list
    #     :param gtruth_file_list: complete gtruth file list
    #     :param iou_thd:
    #     :return:
    #     """
    #
    #     history = []
    #     self._init_metric()
    #     assert len(gtruth_file_list) == len(det_file_list)
    #     num_image = len(gtruth_file_list)
    #     with tqdm(total=num_image) as pbar:
    #         for det_file, gtruth_file in \
    #                 zip(det_file_list, gtruth_file_list):
    #
    #
    #             preds = load_bbox_from_file(det_file)
    #             gts = load_bbox_from_file(gtruth_file)
    #             singe_image_metric = eval_single_image(preds=preds, gts=gts, iou_thd=iou_thd)
    #
    #
    #             self._update_metric(singe_image_metric)
    #             history.append(
    #                 [det_file, gtruth_file] + singe_image_metric
    #             )
    #             pbar.set_description(f'| tp = {self.TP_}, fp = {self.FP_}, fn = {self.FN_}')
    #             # f' avg_iou = {np.average(self.IoUs_)}'
    #             pbar.update(1)
    #     precision, recall, f1_score = self._wrap_result()
    #
    #     return history, precision, recall, f1_score

    # def eval_img_list(self, test_image_path_list, image_root_dir,
    #                   det_root_dir, gt_root_dir, iou_thd=0.5):
    #     det_file_list, gtruth_file_list = get_det_gt_txt_file_list(
    #         test_image_path_list=test_image_path_list,
    #         image_root_dir=image_root_dir,
    #         det_root_dir=det_root_dir,
    #         gt_root_dir=gt_root_dir)
    #
    #     return self._eval(
    #         det_file_list=det_file_list, gtruth_file_list=gtruth_file_list,
    #         iou_thd=iou_thd
    #     )
    #
    # def eval_img_det_gt_data_root_dir(
    #         self, image_root_dir, det_root_dir, gt_root_dir, dir_list=None):
    #     print(f'| Total {len(dir_list)} directories. ')
    #     test_image_path_list = load_image_list(image_root_dir, dir_list=dir_list)
    #
    #     return self.eval_img_list(
    #         test_image_path_list=test_image_path_list,
    #         image_root_dir=image_root_dir,
    #         det_root_dir=det_root_dir,
    #         gt_root_dir=gt_root_dir
    #     )