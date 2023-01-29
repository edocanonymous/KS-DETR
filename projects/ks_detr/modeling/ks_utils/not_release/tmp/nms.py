# from https://mmdetection.readthedocs.io/en/latest/_modules/mmdet/core/post_processing/bbox_nms.html#fast_nms
import torch
from torch import nn as nn
from mmdet.core.post_processing.bbox_nms import multiclass_nms, fast_nms
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from mmcv.ops.nms import batched_nms


# from models.ksgt.scoring_gt import unnormalize_box


class ProposalProcessV0(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=100, min_score=-1) -> None:  # 0.05
        super().__init__()
        self.num_select = num_select
        self.min_score = min_score

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model

                    'pred_logits' = {Tensor: 2} tensor([[[-8.9165, -4.4727, -5.9848,  ..., -5.3282, -6.6623, -6.8260],\n         [-7.9659, -5.0586, -6.6094,  ..., -6.3418, -6.7981, -7.5071],\n         [-8.2270, -5.2689, -6.7123,  ..., -5.1881, -5.9215, -5.8901],\n         ...,\n         [-7.7876, -4.2261, -4.6698,  ..., -4.5377, -5.8694, -4.7226],\n         [-8.1850, -4.6232, -5.6204,  ..., -6.5370, -7.7144, -6.7796],\n         [-8.1698, -5.0259, -5.8377,  ..., -6.6567, -7.3602, -7.3285]],\n\n        [[-8.5812, -5.1892, -5.9446,  ..., -6.3609, -7.4751, -7.0526],\n         [-7.4673, -6.1421, -6.4913,  ..., -5.8209, -7.3903, -6.6696],\n         [-8.2685, -6.1065, -6.4033,  ..., -6.5511, -8.0312, -7.3727],\n         ...,\n         [-8.0465, -6.7189, -7.6001,  ..., -6.9100, -8.3376, -7.2618],\n         [-6.9823, -7.3425, -7.6290,  ..., -4.4328, -7.7821, -5.5195],\n         [-7.3953, -6.5201, -6.6581,  ..., -6.0785, -7.5598, -6.7984]]],\n       device='cuda:0')
                    'pred_boxes' = {Tensor: 2} tensor([[[0.0215, 0.1860, 0.0414, 0.1624],\n         [0.2881, 0.3976, 0.5520, 0.6339],\n         [0.4088, 0.0554, 0.0416, 0.0513],\n         ...,\n         [0.5143, 0.4885, 0.0168, 0.0438],\n         [0.5236, 0.6102, 0.3855, 0.2386],\n         [0.2479, 0.6086, 0
                    'aux_outputs' = {list: 5} [{'pred_logits': tensor([[[-8.3371, -4.3174, -5.7542,  ..., -5.2268, -5.5282, -5.6161],\n         [-7.6300, -6.3398, -6.9885,  ..., -6.9629, -6.3744, -7.2740],\n         [-8.2829, -5.4666, -6.1123,  ..., -5.6119, -5.4080, -6.3546],\n         ...,\n         [-7.0408, -2.9744, -3.7967,  ..., -3.4046, -4.3407, -3.2300],\n         [-7.7358, -6.4145, -6.2673,  ..., -7.4811, -7.4849, -7.1782],\n         [-7.6741, -5.8953, -5.7857,  ..., -6.8710, -6.6873, -6.9215]],\n\n        [[-8.2756, -4.8763, -5.6707,  ..., -5.7167, -6.2823, -6.2206],\n         [-7.6345, -5.7585, -5.3776,  ..., -5.7358, -6.3607, -5.2409],\n         [-8.2626, -5.3800, -5.4123,  ..., -5.7813, -6.5961, -6.1630],\n         ...,\n         [-9.1962, -6.2226, -7.2094,  ..., -7.0788, -7.9134, -6.2083],\n         [-7.0043, -3.7835, -4.0759,  ..., -2.9114, -5.7502, -4.0333],\n         [-8.7625, -6.6498, -6.7359,  ..., -6.3321, -6.7741, -5.8506]]],\n       device='cuda:0'), 'pred_boxes': tensor([[[0.0143, 0.1764, 0.0288, 0.0765],\n         [0.2591,...
                    'ksgt_output_list' = {list: 0} []
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # assert len(out_logits) == len(target_sizes)
        # assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))  # torch.Size([2, 100, 4])
        # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        #
        # # and from relative [0, 1] to absolute [0, height] coordinates
        # img_h, img_w = target_sizes.unbind(1)
        # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # boxes = boxes * scale_fct[:, None, :]
        results = []
        for s, l, b in zip(scores, labels, boxes):
            inds = s > self.min_score
            b = b[inds, :]
            l = l[inds]
            results.append({'scores': s, 'labels': l, 'boxes': b})
        # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


def proposal_nms(boxes, scores, labels, score_thr,
                 nms_cfg,
                 max_num=-1,
                 score_factors=None,
                 return_inds=False):
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # # multiply score_factor after threshold to preserve more bboxes, improve
    # # mAP by 1% for YOLOv3
    # if score_factors is not None:
    #     # expand the shape to match original shape of score
    #     score_factors = score_factors.view(-1, 1).expand(
    #         multi_scores.size(0), num_classes)
    #     score_factors = score_factors.reshape(-1)
    #     scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        boxes = torch.cat([boxes, boxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if boxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        dets = torch.cat([boxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    dets, keep = batched_nms(boxes, scores, labels, nms_cfg=nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], inds[keep]
    else:
        return dets, labels[keep]


class ProposalProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api
    Faster RCNN:
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
    """

    # num_select = 100, min_score = 0.05, nms_thd = 0.5
    # **kwargs  num_select=None, min_score=None, nms_thd=None
    # num_select, min_score, nms_thd = kwargs.pop('num_select', None), kwargs.pop('min_score', None),\
    #                                  kwargs.pop('nms_thd', None)
    def __init__(self, num_select=None, min_score=None, nms_thd=None) -> None:

        super().__init__()
        self.num_select = num_select if num_select is not None else 100
        self.min_score = min_score if min_score is not None else 0.0
        self.nms_thd = nms_thd if nms_thd is not None else 1.0
        self.nms_cfg = {'class_agnostic': True, 'iou_threshold': self.nms_thd}  # , 'max_num': self.num_select

    def bbox_filtering(self, boxes, scores, labels):
        results = []
        for score, pred_box, label in zip(scores, boxes, labels):
            # bbox to x1, y1, x2 ,y2 format.

            # Returns: tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            #             (k), and (k). Dets are boxes with scores. Labels are 0-based.
            #  make sure the last column is the background score to correctly use multiclass_nms.
            # 'pred_logits' : torch.Size([2, 300, 91]), the last one is indeed the background class.
            dets, labels = proposal_nms(boxes=pred_box, scores=score,
                                        labels=label, score_thr=self.min_score,
                                        nms_cfg=self.nms_cfg, max_num=self.num_select
                                        )
            b, s = dets[:, :4], dets[:, 4:5]
            results.append({'scores': s, 'labels': labels, 'boxes': b})
        # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model

                    'pred_logits' = {Tensor: 2} tensor([[[-8.9165, -4.4727, -5.9848,  ..., -5.3282, -6.6623, -6.8260],\n         [-7.9659, -5.0586, -6.6094,  ..., -6.3418, -6.7981, -7.5071],\n         [-8.2270, -5.2689, -6.7123,  ..., -5.1881, -5.9215, -5.8901],\n         ...,\n         [-7.7876, -4.2261, -4.6698,  ..., -4.5377, -5.8694, -4.7226],\n         [-8.1850, -4.6232, -5.6204,  ..., -6.5370, -7.7144, -6.7796],\n         [-8.1698, -5.0259, -5.8377,  ..., -6.6567, -7.3602, -7.3285]],\n\n        [[-8.5812, -5.1892, -5.9446,  ..., -6.3609, -7.4751, -7.0526],\n         [-7.4673, -6.1421, -6.4913,  ..., -5.8209, -7.3903, -6.6696],\n         [-8.2685, -6.1065, -6.4033,  ..., -6.5511, -8.0312, -7.3727],\n         ...,\n         [-8.0465, -6.7189, -7.6001,  ..., -6.9100, -8.3376, -7.2618],\n         [-6.9823, -7.3425, -7.6290,  ..., -4.4328, -7.7821, -5.5195],\n         [-7.3953, -6.5201, -6.6581,  ..., -6.0785, -7.5598, -6.7984]]],\n       device='cuda:0')
                    'pred_boxes' = {Tensor: 2} tensor([[[0.0215, 0.1860, 0.0414, 0.1624],\n         [0.2881, 0.3976, 0.5520, 0.6339],\n         [0.4088, 0.0554, 0.0416, 0.0513],\n         ...,\n         [0.5143, 0.4885, 0.0168, 0.0438],\n         [0.5236, 0.6102, 0.3855, 0.2386],\n         [0.2479, 0.6086, 0
                    'aux_outputs' = {list: 5} [{'pred_logits': tensor([[[-8.3371, -4.3174, -5.7542,  ..., -5.2268, -5.5282, -5.6161],\n         [-7.6300, -6.3398, -6.9885,  ..., -6.9629, -6.3744, -7.2740],\n         [-8.2829, -5.4666, -6.1123,  ..., -5.6119, -5.4080, -6.3546],\n         ...,\n         [-7.0408, -2.9744, -3.7967,  ..., -3.4046, -4.3407, -3.2300],\n         [-7.7358, -6.4145, -6.2673,  ..., -7.4811, -7.4849, -7.1782],\n         [-7.6741, -5.8953, -5.7857,  ..., -6.8710, -6.6873, -6.9215]],\n\n        [[-8.2756, -4.8763, -5.6707,  ..., -5.7167, -6.2823, -6.2206],\n         [-7.6345, -5.7585, -5.3776,  ..., -5.7358, -6.3607, -5.2409],\n         [-8.2626, -5.3800, -5.4123,  ..., -5.7813, -6.5961, -6.1630],\n         ...,\n         [-9.1962, -6.2226, -7.2094,  ..., -7.0788, -7.9134, -6.2083],\n         [-7.0043, -3.7835, -4.0759,  ..., -2.9114, -5.7502, -4.0333],\n         [-8.7625, -6.6498, -6.7359,  ..., -6.3321, -6.7741, -5.8506]]],\n       device='cuda:0'), 'pred_boxes': tensor([[[0.0143, 0.1764, 0.0288, 0.0765],\n         [0.2591,...
                    'ksgt_output_list' = {list: 0} []
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # assert len(out_logits) == len(target_sizes)
        # assert target_sizes.shape[1] == 2

        scores = out_logits.sigmoid()

        boxes = box_cxcywh_to_xyxy(out_bbox)
        # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        #
        # # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        # boxes = box_cxcywh_to_xyxy(out_bbox)
        # box_unnormalized = unnormalize_box(box_normalized=out_bbox,
        #                                    input_img_size=)  # img_target['size']
        # return self.box_filtering(boxes=boxes, scores=scores)

        results = []
        for score, pred_box in zip(scores, boxes):
            # bbox to x1, y1, x2 ,y2 format.

            # Returns: tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            #             (k), and (k). Dets are boxes with scores. Labels are 0-based.
            #  make sure the last column is the background score to correctly use multiclass_nms.
            # 'pred_logits' : torch.Size([2, 300, 91]), the last one is indeed the background class.

            # box_unnormalized = unnormalize_box(box_normalized=out_bbox,
            #                                    input_img_size=)  # img_target['size']


            dets, labels = multiclass_nms(
                multi_bboxes=pred_box,
                multi_scores=score,
                score_thr=self.min_score,
                nms_cfg=self.nms_cfg,
                max_num=self.num_select,
                # return_inds=True
            )
            # TODO: if necessary, map the box to normalized box.
            b, s = dets[:, :4], dets[:, 4:5]
            results.append({'scores': s, 'labels': labels, 'boxes': b})
        # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results

    def top_proposals(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model

                    'pred_logits' = {Tensor: 2} tensor([[[-8.9165, -4.4727, -5.9848,  ..., -5.3282, -6.6623, -6.8260],\n         [-7.9659, -5.0586, -6.6094,  ..., -6.3418, -6.7981, -7.5071],\n         [-8.2270, -5.2689, -6.7123,  ..., -5.1881, -5.9215, -5.8901],\n         ...,\n         [-7.7876, -4.2261, -4.6698,  ..., -4.5377, -5.8694, -4.7226],\n         [-8.1850, -4.6232, -5.6204,  ..., -6.5370, -7.7144, -6.7796],\n         [-8.1698, -5.0259, -5.8377,  ..., -6.6567, -7.3602, -7.3285]],\n\n        [[-8.5812, -5.1892, -5.9446,  ..., -6.3609, -7.4751, -7.0526],\n         [-7.4673, -6.1421, -6.4913,  ..., -5.8209, -7.3903, -6.6696],\n         [-8.2685, -6.1065, -6.4033,  ..., -6.5511, -8.0312, -7.3727],\n         ...,\n         [-8.0465, -6.7189, -7.6001,  ..., -6.9100, -8.3376, -7.2618],\n         [-6.9823, -7.3425, -7.6290,  ..., -4.4328, -7.7821, -5.5195],\n         [-7.3953, -6.5201, -6.6581,  ..., -6.0785, -7.5598, -6.7984]]],\n       device='cuda:0')
                    'pred_boxes' = {Tensor: 2} tensor([[[0.0215, 0.1860, 0.0414, 0.1624],\n         [0.2881, 0.3976, 0.5520, 0.6339],\n         [0.4088, 0.0554, 0.0416, 0.0513],\n         ...,\n         [0.5143, 0.4885, 0.0168, 0.0438],\n         [0.5236, 0.6102, 0.3855, 0.2386],\n         [0.2479, 0.6086, 0
                    'aux_outputs' = {list: 5} [{'pred_logits': tensor([[[-8.3371, -4.3174, -5.7542,  ..., -5.2268, -5.5282, -5.6161],\n         [-7.6300, -6.3398, -6.9885,  ..., -6.9629, -6.3744, -7.2740],\n         [-8.2829, -5.4666, -6.1123,  ..., -5.6119, -5.4080, -6.3546],\n         ...,\n         [-7.0408, -2.9744, -3.7967,  ..., -3.4046, -4.3407, -3.2300],\n         [-7.7358, -6.4145, -6.2673,  ..., -7.4811, -7.4849, -7.1782],\n         [-7.6741, -5.8953, -5.7857,  ..., -6.8710, -6.6873, -6.9215]],\n\n        [[-8.2756, -4.8763, -5.6707,  ..., -5.7167, -6.2823, -6.2206],\n         [-7.6345, -5.7585, -5.3776,  ..., -5.7358, -6.3607, -5.2409],\n         [-8.2626, -5.3800, -5.4123,  ..., -5.7813, -6.5961, -6.1630],\n         ...,\n         [-9.1962, -6.2226, -7.2094,  ..., -7.0788, -7.9134, -6.2083],\n         [-7.0043, -3.7835, -4.0759,  ..., -2.9114, -5.7502, -4.0333],\n         [-8.7625, -6.6498, -6.7359,  ..., -6.3321, -6.7741, -5.8506]]],\n       device='cuda:0'), 'pred_boxes': tensor([[[0.0143, 0.1764, 0.0288, 0.0765],\n         [0.2591,...
                    'ksgt_output_list' = {list: 0} []
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # assert len(out_logits) == len(target_sizes)
        # assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))  # torch.Size([2, 100, 4])
        # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        #
        # # and from relative [0, 1] to absolute [0, height] coordinates
        # img_h, img_w = target_sizes.unbind(1)
        # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # boxes = boxes * scale_fct[:, None, :]
        results = []
        for s, l, b in zip(scores, labels, boxes):
            inds = s > self.min_score
            b = b[inds, :]
            l = l[inds]
            results.append({'scores': s, 'labels': l, 'boxes': b})
        # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results
