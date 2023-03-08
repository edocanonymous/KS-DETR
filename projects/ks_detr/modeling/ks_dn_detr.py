from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.utils.misc import inverse_sigmoid
from detectron2.modeling import detector_postprocess

from projects.dn_detr.modeling import DNDETR
from projects.ks_detr.modeling.ks_utils import set_ksgt_target, prepare_ksgt_targets, set_ksgt_inference_output


class KSDNDETR(DNDETR):
    """Implement DN-DETR in `DN-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2201.12329>`_

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
        freeze_anchor_box_centers (bool): If True, freeze the center param ``(x, y)`` for
            the initialized dynamic anchor boxes in format ``(x, y, w, h)``
            and only train ``(w, h)``. Default: True.
        select_box_nums_for_evaluation (int): Select the top-k confidence predicted boxes for inference.
            Default: 300.
        denoising_groups (int): Number of groups for noised ground truths. Default: 5.
        label_noise_prob (float): The probability of the label being noised. Default: 0.2.
        box_noise_scale (float): Scaling factor for box noising. Default: 0.4.
        with_indicator (bool): If True, add indicator in denoising queries part and matching queries part.
            Default: True.
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
            freeze_anchor_box_centers: bool = True,
            select_box_nums_for_evaluation: int = 300,
            denoising_groups: int = 5,
            label_noise_prob: float = 0.2,
            box_noise_scale: float = 0.4,
            with_indicator: bool = True,
            device="cuda",

            ksgt: nn.Module = None,
    ):
        super().__init__(
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
            freeze_anchor_box_centers=freeze_anchor_box_centers,
            select_box_nums_for_evaluation=select_box_nums_for_evaluation,
            denoising_groups=denoising_groups,
            label_noise_prob=label_noise_prob,
            box_noise_scale=box_noise_scale,
            with_indicator=with_indicator,
            device=device,
        )
        self.ksgt = ksgt

    def forward(self, batched_inputs):
        """Forward function of `DN-DETR` which excepts a list of dict as inputs.

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
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)
        # torch.Size([2, 3, 672, 909])  image_sizes = [(672, 909), (640, 853)]

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

        # only use last level feature for DAB-DETR
        features = self.backbone(images.tensor)[self.in_features[-1]]  # torch.Size([2, 2048, 21, 29])
        features = self.input_proj(features)  # torch.Size([2, 256, 21, 29])
        img_masks = F.interpolate(img_masks[None], size=features.shape[-2:]).to(torch.bool)[
            0]  # torch.Size([2, 21, 29])
        pos_embed = self.position_embedding(img_masks)  # torch.Size([2, 256, 21, 29])

        # collect ground truth for denoising generation
        # if self.training:
        #     gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        #     targets = self.prepare_targets(gt_instances)
        #     gt_labels_list = [t["labels"] for t in targets]
        #     gt_boxes_list = [t["boxes"] for t in targets]
        # else:
        #     # set to None during inference
        #     targets = None

        # ------------------------------
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets = self.prepare_targets(gt_instances)
        set_ksgt_target(ksgt=self.ksgt, img_masks=img_masks, features=features, targets=targets,
                        padded_img_size=(H, W))
        # ------------------------------
        if self.training:
            gt_labels_list = [t["labels"] for t in targets]
            gt_boxes_list = [t["boxes"] for t in targets]
        else:
            # set to None during inference
            targets = None

        # for vallina dn-detr, label queries in the matching part is encoded as "no object" (the last class)
        # in the label encoder.
        matching_label_query = self.denoising_generator.label_encoder(
            torch.tensor(self.num_classes).to(self.device)
        ).repeat(self.num_queries, 1)
        indicator_for_matching_part = torch.zeros([self.num_queries, 1]).to(self.device)
        matching_label_query = torch.cat(
            [matching_label_query, indicator_for_matching_part], 1
        ).repeat(batch_size, 1, 1)
        matching_box_query = self.anchor_box_embed.weight.repeat(batch_size, 1, 1)

        if targets is None:
            input_label_query = matching_label_query.transpose(0, 1)  # (num_queries, bs, embed_dim)
            input_box_query = matching_box_query.transpose(0, 1)  # (num_queries, bs, 4)
            attn_mask = None
            denoising_groups = self.denoising_groups
            max_gt_num_per_image = 0
        else:
            # generate denoising queries and attention masks
            (
                noised_label_queries,
                noised_box_queries,
                attn_mask,
                denoising_groups,
                max_gt_num_per_image,
            ) = self.denoising_generator(gt_labels_list, gt_boxes_list)

            # concate dn queries and matching queries as input
            input_label_query = torch.cat(
                [noised_label_queries, matching_label_query], 1
            ).transpose(0, 1)
            input_box_query = torch.cat([noised_box_queries, matching_box_query], 1).transpose(0, 1)

        hidden_states, reference_boxes, *intermediate_output_dict = self.transformer(
            features,
            img_masks,
            input_box_query,
            pos_embed,
            target=input_label_query,
            attn_mask=[attn_mask, None],  # None mask for cross attention

            ksgt=self.ksgt,
        )

        # Calculate output coordinates and classes.
        reference_boxes = inverse_sigmoid(reference_boxes)
        anchor_box_offsets = self.bbox_embed(hidden_states)
        outputs_coord = (reference_boxes + anchor_box_offsets).sigmoid()
        outputs_class = self.class_embed(hidden_states)

        # denoising post process
        output = {
            "denoising_groups": torch.tensor(denoising_groups).to(self.device),
            "max_gt_num_per_image": torch.tensor(max_gt_num_per_image).to(self.device),
        }
        outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, output)

        output.update({"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]})
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.training:
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls, box_pred = set_ksgt_inference_output(
                output=output, ksgt=self.ksgt, aux_loss=self.aux_loss)

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
        #     # new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        #     new_targets.append(
        #         {"labels": gt_classes,
        #          "boxes": gt_boxes,
        #          # ================
        #          "box_unnormalized": targets_per_image.gt_boxes.tensor,
        #          'image_size': torch.tensor(targets_per_image.image_size, device=self.device),
        #          # input image size  (544, 658), a tuple in cpu to tensor in cuda
        #          }
        #     )
        # return new_targets

