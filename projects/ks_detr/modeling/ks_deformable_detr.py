# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances

from projects.deformable_detr.modeling import DeformableDETR
from projects.ks_detr.modeling.detr_module_utils import set_ksgt_multi_scale_target, prepare_ksgt_targets, \
    set_ksgt_inference_output


class KSDeformableDETR(DeformableDETR):
    """Implements the Deformable DETR model.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        backbone (nn.Module): the backbone module.
        position_embedding (nn.Module): the position embedding module.
        neck (nn.Module): the neck module.
        transformer (nn.Module): the transformer module.
        embed_dim (int): the dimension of the embedding.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): whether to use auxiliary loss. Default: True.
        with_box_refine (bool): whether to use box refinement. Default: False.
        as_two_stage (bool): whether to use two-stage. Default: False.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 100.

    """

    def __init__(
            self,
            backbone,
            position_embedding,
            neck,
            transformer,
            embed_dim,
            num_classes,
            num_queries,
            criterion,
            pixel_mean,
            pixel_std,
            aux_loss=True,
            with_box_refine=False,
            as_two_stage=False,
            select_box_nums_for_evaluation=100,
            device="cuda",

            ksgt: nn.Module = None,
    ):
        super().__init__(
            backbone=backbone,
            position_embedding=position_embedding,
            neck=neck,
            transformer=transformer,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            criterion=criterion,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            aux_loss=aux_loss,
            with_box_refine=with_box_refine,
            as_two_stage=as_two_stage,
            select_box_nums_for_evaluation=select_box_nums_for_evaluation,
            device=device,
        )
        self.ksgt = ksgt
        # self.ksgt.init_weights()

    def _extract_predict(self, inter_states, init_reference, inter_references):
        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        # Note: self.class_embed, self.bbox_embed = nn.ModuleList, nn.ModuleList
        for lvl in range(inter_states.shape[0]):  # 0, -> 6
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)  # torch.Size([2, 300, 80])
            outputs_coords.append(outputs_coord)  # torch.Size([2, 300, 4])

        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]
        return outputs_class, outputs_coord

    def extract_output(self, inter_states, init_reference, inter_references, ksgt=None):
        # Note: self.class_embed, self.bbox_embed = nn.ModuleList, nn.ModuleList
        #  need to handle multiple class_embed, bbox_embed in the output of the transformer.
        # init_reference is always a tensor, and the decoder output share the same init_reference
        if isinstance(inter_states, list) and isinstance(inter_references, list):
            outputs_class = []
            outputs_coord = []

            # ksgt = kwargs.get('ksgt', None)
            teacher_attn_return_no_intermediate_out = True \
                if ksgt and ksgt.teacher_attn_return_no_intermediate_out else False

            for k, (single_decoder_inter_state, single_decoder_inter_references) \
                    in enumerate(zip(inter_states, inter_references)):
                outputs_classes_single, outputs_coords_single = self._extract_predict(
                    inter_states=single_decoder_inter_state,
                    init_reference=init_reference,
                    inter_references=single_decoder_inter_references)
                # The first output is from the decoder of the normal attention.
                # The output of teacher attentions are from the second one.
                if k > 0 and teacher_attn_return_no_intermediate_out:
                    # Extract the output of the last layer only and keep the dimension.
                    outputs_class.append(outputs_classes_single[-1].unsqueeze(0))
                    outputs_coord.append(outputs_coords_single[-1].unsqueeze(0))
                else:
                    outputs_class.append(outputs_classes_single)
                    outputs_coord.append(outputs_coords_single)

            outputs_class, outputs_coord = torch.concat(outputs_class, dim=0), \
                                           torch.concat(outputs_coord, dim=0)
        else:
            assert torch.is_tensor(inter_states) and torch.is_tensor(init_reference) and torch.is_tensor(
                inter_references)
            outputs_class, outputs_coord = self._extract_predict(
                inter_states=inter_states, init_reference=init_reference,
                inter_references=inter_references)

        return outputs_class, outputs_coord

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)

        # https://github.com/IDEA-Research/detrex/issues/134
        # if self.training:
        #     batch_size, _, H, W = images.tensor.shape
        #     img_masks = images.tensor.new_ones(batch_size, H, W)
        #     for img_id in range(batch_size):
        #         # mask padding regions in batched images
        #         img_h, img_w = batched_inputs[img_id]["instances"].image_size
        #         img_masks[img_id, :img_h, :img_w] = 0
        # else:
        #     batch_size, _, H, W = images.tensor.shape
        #     img_masks = images.tensor.new_zeros(batch_size, H, W)

        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = images.image_sizes[img_id]
                img_masks[img_id, :(img_h - 1), :(img_w - 1)] = 0

        # original features
        features = self.backbone(images.tensor)  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in deformable DETR
        multi_level_feats = self.neck(features)  # a list of 4 elements
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )  # torch.Size([2, 80, 102])
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # ------------------------------
        # We always need gt boxes during training and inference for teacher branch,
        # student branch do not need gt boxes during inference
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets = self.prepare_targets(gt_instances)
        # TODO: check if self.ksgt is modified inplace
        # ksgt_set_target(ksgt=self.ksgt, img_masks=img_masks, features=features, targets=targets,
        #                 padded_img_size=(H, W))
        set_ksgt_multi_scale_target(
            ksgt=self.ksgt, img_masks=multi_level_masks,
            features=multi_level_feats, targets=targets,
            padded_img_size=(H, W))
        # ============================

        # initialize object query embeddings
        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact, *intermediate_output_dict
        ) = self.transformer(
            multi_level_feats, multi_level_masks, multi_level_position_embeddings, query_embeds,

            ksgt=self.ksgt,
        )
        outputs_class, outputs_coord = self.extract_output(
            inter_states=inter_states, init_reference=init_reference,
            inter_references=inter_references,
            ksgt=self.ksgt  # will be use for handling the output of multi-head transformer
        )
        # # Calculate output coordinates and classes.
        # outputs_classes = []
        # outputs_coords = []

        # for lvl in range(inter_states.shape[0]):  # 0, -> 6
        #     if lvl == 0:
        #         reference = init_reference
        #     else:
        #         reference = inter_references[lvl - 1]
        #     reference = inverse_sigmoid(reference)
        #     outputs_class = self.class_embed[lvl](inter_states[lvl])
        #     tmp = self.bbox_embed[lvl](inter_states[lvl])
        #     if reference.shape[-1] == 4:
        #         tmp += reference
        #     else:
        #         assert reference.shape[-1] == 2
        #         tmp[..., :2] += reference
        #     outputs_coord = tmp.sigmoid()
        #     outputs_classes.append(outputs_class)
        #     outputs_coords.append(outputs_coord)

        # outputs_class = torch.stack(outputs_classes)
        # # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        # outputs_coord = torch.stack(outputs_coords)
        # # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        # TODO: do we need to calculate the enc_outputs loss for the teacher attention?
        if self.as_two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            output["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            # ===============================
            # box_cls = output["pred_logits"]
            # box_pred = output["pred_boxes"]
            box_cls, box_pred = set_ksgt_inference_output(
                output=output, ksgt=self.ksgt, aux_loss=self.aux_loss)
            # ===============================

            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def prepare_targets(self, targets):
        return prepare_ksgt_targets(targets=targets, device=self.device)

        # new_targets = []
        # for targets_per_image in targets:
        #     h, w = targets_per_image.image_size
        #     image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
        #     gt_classes = targets_per_image.gt_classes
        #     gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
        #     gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
        #     new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        # return new_targets
