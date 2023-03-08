import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.utils import inverse_sigmoid
from detectron2.modeling import detector_postprocess
from projects.dn_deformable_detr.modeling import DNDeformableDETR
from projects.ks_detr.modeling.ks_utils import set_ksgt_multi_scale_target, prepare_ksgt_targets, \
    set_ksgt_inference_output


class KSDNDeformableDETR(DNDeformableDETR):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.01305>`_.
    Code is modified from the `official github repo
    <https://github.com/IDEA-opensource/DN-DETR>`_.
    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 100.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        backbone,
        position_embedding,
        neck,
        transformer,
        num_classes,
        num_queries,
        criterion,
        pixel_mean,
        pixel_std,
        embed_dim=256,
        aux_loss=True,
        as_two_stage=False,
        denoising_groups: int = 5,
        label_noise_prob: float = 0.2,
        box_noise_scale: float = 0.4,
        with_indicator: bool = True,
        select_box_nums_for_evaluation: int = 300,
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
            select_box_nums_for_evaluation=select_box_nums_for_evaluation,
            device=device,
            as_two_stage=as_two_stage,
            denoising_groups=denoising_groups,
            label_noise_prob=label_noise_prob,
            with_indicator=with_indicator,
            box_noise_scale=box_noise_scale,

        )
        self.ksgt = ksgt

    def _extract_predict(self, inter_states, init_reference, inter_references):
        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
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
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
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
        """Forward function of `DN-Deformable-DETR` which excepts a list of dict as inputs.
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

        # original features
        features = self.backbone(images.tensor)

        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []

        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # ------------------------------
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets = self.prepare_targets(gt_instances)
        set_ksgt_multi_scale_target(
            ksgt=self.ksgt, img_masks=multi_level_masks,
            features=multi_level_feats, targets=targets,
            padded_img_size=(H, W))
        # ------------------------------

        # collect ground truth for denoising generation
        if self.training:
            # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # targets = self.prepare_targets(gt_instances)
            gt_labels_list = [t["labels"] for t in targets]
            gt_boxes_list = [t["boxes"] for t in targets]
        else:
            # set to None during inference
            targets = None

        matching_label_query = self.tgt_embed.weight
        # add indicator in the last dimension if needed
        if self.with_indicator:
            indicator_for_matching_part = torch.zeros([self.num_queries, 1]).to(self.device)
            matching_label_query = torch.cat([matching_label_query, indicator_for_matching_part], 1)
        matching_label_query = matching_label_query.repeat(batch_size, 1, 1)
        matching_box_query = self.refpoint_embed.weight.repeat(batch_size, 1, 1)

        if targets is None:
            input_label_query = matching_label_query  # (num_queries, bs, embed_dim)
            input_box_query = matching_box_query  # (num_queries, bs, 4)
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
            input_label_query = torch.cat([noised_label_queries, matching_label_query], 1)
            input_box_query = torch.cat([noised_box_queries, matching_box_query], 1)

        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,  # [0..1]
            *intermediate_output_dict
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            input_label_query,
            input_box_query,
            attn_masks=[attn_mask, None],

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
        # for lvl in range(inter_states.shape[0]):
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

        # denoising post process
        output = {
            "denoising_groups": torch.tensor(denoising_groups).to(self.device),
            "max_gt_num_per_image": torch.tensor(max_gt_num_per_image).to(self.device),
        }
        outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, output)

        output.update({"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]})
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        if self.as_two_stage:
            interm_coord = enc_reference
            interm_class = self.class_embed[-1](enc_state)
            output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}

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
