from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.utils.misc import inverse_sigmoid
from detectron2.modeling import detector_postprocess
from projects.conditional_detr.modeling import ConditionalDETR
from projects.ks_detr.modeling.ks_utils import set_ksgt_target, prepare_ksgt_targets, set_ksgt_inference_output


class KSConditionalDETR(ConditionalDETR):
    """Implement Conditional-DETR in `Conditional DETR for Fast Training Convergence
    <https://arxiv.org/abs/2108.06152>`_

    Args:
        backbone (nn.Module): Backbone module for feature extraction.
        in_features (List[str]): Selected backbone output features for transformer module.
        in_channels (int): Dimension of the last feature in `in_features`.
        position_embedding (nn.Module): Position encoding layer for generating position embeddings.
        transformer (nn.Module): Transformer module used for further processing features and input queries.
        embed_dim (int): Hidden dimension for transformer module.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        select_box_nums_for_evaluation (int): Select the top-k confidence predicted boxes for inference.
            Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: List[str],
        in_channels: int,
        position_embedding: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: nn.Module,
        aux_loss: bool = True,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        select_box_nums_for_evaluation: int = 300,
        device: str = "cuda",

        ksgt: nn.Module = None,
    ):
        super(KSConditionalDETR, self).__init__(
            backbone=backbone,
            in_features=in_features,
            in_channels=in_channels,
            position_embedding=position_embedding,
            transformer=transformer,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            criterion=criterion,
            aux_loss=aux_loss,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            select_box_nums_for_evaluation=select_box_nums_for_evaluation,
            device=device,
        )
        self.ksgt = ksgt

    def forward(self, batched_inputs):
        """Forward function of `DAB-DETR` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries.
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)

        # if self.training:
        #     batch_size, _, H, W = images.tensor.shape
        #     img_masks = images.tensor.new_ones(batch_size, H, W)
        #     for img_id in range(batch_size):
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

        # only use last level feature in Conditional-DETR
        features = self.backbone(images.tensor)[self.in_features[-1]]
        features = self.input_proj(features)
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.position_embedding(img_masks)

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets = self.prepare_targets(gt_instances)
        set_ksgt_target(ksgt=self.ksgt, img_masks=img_masks, features=features, targets=targets,
                        padded_img_size=(H, W))

        hidden_states, reference, *intermediate_output_dict = self.transformer(
            features, img_masks, self.query_embed.weight, pos_embed,
            ksgt=self.ksgt,
        )

        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(hidden_states.shape[0]):
            tmp = self.bbox_embed(hidden_states[lvl])
            tmp[..., :2] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)
        outputs_class = self.class_embed(hidden_states)

        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

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

