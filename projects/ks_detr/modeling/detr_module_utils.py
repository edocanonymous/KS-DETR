from detrex.layers import box_xyxy_to_cxcywh

import torch


# TODO: update decoder configuration.

def set_ksgt_target(ksgt, img_masks, features, targets, padded_img_size):
    if ksgt:  # prepare fg bg mask if necessary
        # (B, H, W), [2, 21, 29]; to change to B, N use .flatten(1)
        ksgt.src_key_padding_mask = img_masks

        bs, c, h, w = features.shape  # torch.Size([2, 256, 25, 32])
        ksgt.feat_map_size = (h, w)
        # self.ksgt.targets = targets

        ksgt.set_ksgt_targets(targets, feat_map_size=(h, w), padded_img_size=padded_img_size)

        # if self.training:
        #     gt_ratio_or_sigma = ksgt_args[1]
        #     self.ksgt.set_gt_ratio_or_sigma(gt_ratio_or_sigma)


def set_ksgt_mask(mask, **kwargs,):
    ksgt = kwargs.get('ksgt', None)
    if ksgt:
        ksgt.mask = mask


def set_ksgt_multi_scale_target(ksgt, img_masks, features, targets, padded_img_size):
    if ksgt:  # prepare fg bg mask if necessary
        # (B, H, W), [2, 21, 29]; to change to B, N use .flatten(1)
        ksgt.src_key_padding_mask = img_masks

        if torch.is_tensor(features):
            bs, c, h, w = features.shape  # torch.Size([2, 256, 25, 32])
            feat_map_size = (h, w)
        else:
            assert isinstance(features, (list, tuple))
            feat_map_size = []
            for feat in features:
                bs, c, h, w = feat.shape  # torch.Size([2, 256, 25, 32])
                feat_map_size.append((h, w))

        # The following two variables are set in set_ksgt_targets
        # ksgt.feat_map_size = (h, w)
        # self.ksgt.targets = targets

        ksgt.set_ksgt_targets(targets, feat_map_size=feat_map_size, padded_img_size=padded_img_size)

        # if self.training:
        #     gt_ratio_or_sigma = ksgt_args[1]
        #     self.ksgt.set_gt_ratio_or_sigma(gt_ratio_or_sigma)


def prepare_ksgt_targets(targets, device):
    new_targets = []
    for targets_per_image in targets:
        h, w = targets_per_image.image_size
        image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=device)
        gt_classes = targets_per_image.gt_classes
        gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
        gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
        # new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        new_targets.append(
            {"labels": gt_classes,
             "boxes": gt_boxes,
             # ================
             "box_unnormalized": targets_per_image.gt_boxes.tensor,
             'image_size': torch.tensor(targets_per_image.image_size, device=device),
             # input image size  (544, 658), a tuple in cpu to tensor in cuda
             }
        )
    return new_targets


def set_ksgt_inference_output(output, ksgt, aux_loss, ):
    box_cls = output["pred_logits"]
    box_pred = output["pred_boxes"]

    # update the decoder layers to evaluate if specified
    if ksgt and ksgt.eval_decoder_layer != -1:  # specify which decoder layer to evaluate
        eval_decoder_layer = ksgt.eval_decoder_layer
        if aux_loss and eval_decoder_layer < len(output["aux_outputs"]):
            box_cls = output["aux_outputs"][eval_decoder_layer]['pred_logits']
            box_pred = output["aux_outputs"][eval_decoder_layer]['pred_boxes']
    return box_cls, box_pred


def extend_student_teacher_decoder_output(hs, references, hs_t,  references_t):
    # For the case self.bbox_embed is a ModuleList (self.bbox_embed[lvl](hs[lvl])), I cannot put everything
    # into a single tensor.
    if isinstance(hs, list) and isinstance(references, list):
        return hs + [hs_t], references + [references_t]
    else:
        assert torch.is_tensor(hs) and torch.is_tensor(references)
        return [hs, hs_t], [references, references_t]


def concat_student_teacher_decoder_output(
        hs, references, references_t, hs_t,
        teacher_attn_return_no_intermediate_out
):
    # For the case self.bbox_embed is a MLP
    if teacher_attn_return_no_intermediate_out:
        # only return the last layer
        hs, references = torch.cat([hs, hs_t[-1:]], dim=0), \
                         torch.cat([references, references_t[-1:]], dim=0)
    else:
        hs, references = torch.cat([hs, hs_t], dim=0), torch.cat([references, references_t], dim=0)
    return hs, references


def extract_optional_output(intermediate_output):
    """Extract optional output from the return like *intermediate_output.
        intermediate_output can be None if no additional return, or
        a list.
        query, *intermediate_output = layer(
                query,
                key,
                value,
                query_pos=query_pos * position_scales,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
    Args:
        intermediate_output:

    Returns:
    """
    # # The return is a list of N item, where N is the number of returning variables.
    # # If there is only one variable, the return is a list of one element. So we need to extract the item.
    if intermediate_output:
        intermediate_output = intermediate_output[0]
    return intermediate_output
