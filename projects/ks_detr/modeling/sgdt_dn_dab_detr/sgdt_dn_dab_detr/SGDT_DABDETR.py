from models.DN_DAB_DETR.DABDETR import PostProcess, MLP, sigmoid_focal_loss, DABDETR, SetCriterion as SetCriterionOld

import math
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from models.DN_DAB_DETR.backbone import build_backbone
from models.DN_DAB_DETR.matcher import build_matcher
from models.DN_DAB_DETR.segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                                             dice_loss)

from models.DN_DAB_DETR.dn_components import prepare_for_dn, dn_post_process, compute_dn_loss

# ==================================== modification
from .transformer import build_transformer
from models.sgdt.sgdt_components import get_num_sgdt_layer
from ..sgdt.sgdt_ import build_sgdt
from models.sgdt.scoring_visualize import VisualizeToken
from models.sgdt.sgdt_components import split_encoder_layer
from models.sgdt.token_scoring import sgdt_token2feat, sgdt_feat2token
from models.sgdt.scoring_gt import unnormalize_box
from models.sgdt_dn_dab_detr.sgdt_fg_bg_classifier import SetCriterion as TokenFgBgClassificationLoss
from models.sgdt.sgdt_components import get_num_of_layer


class SGDT_DABDETR(DABDETR):
    """ This is the DAB-DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, iter_update=True, query_dim=4,
                 bbox_embed_diff_each_layer=False, random_refpoints_xy=False,
                 # -------------
                 sgdt=None,
                 # sgdt_module=None,
                 train_token_scoring_only=False
                 ):

        super().__init__(backbone=backbone, transformer=transformer, num_classes=num_classes,
                         num_queries=num_queries, aux_loss=aux_loss, iter_update=iter_update, query_dim=query_dim,
                         bbox_embed_diff_each_layer=bbox_embed_diff_each_layer, random_refpoints_xy=random_refpoints_xy)

        self.sgdt = sgdt
        # self.sgdt_module = sgdt_module
        self.train_token_scoring_only = train_token_scoring_only

        # two decoder
        if self.iter_update and sgdt.double_head_transformer:
            self.transformer.decoder_t.bbox_embed = self.bbox_embed

    def forward(self, samples: NestedTensor, dn_args=None, sgdt_args=None,
                teacher_encoder_output_list=None,
                training_skip_forward_decoder=False,
                # proposal_processor=None,
                ):
        """
            Add two functions prepare_for_dn and dn_post_process to implement dn
            The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # # pos? learnable positional embedding, or fixed sin position?  PositionEmbeddingSineHW(), fixed sin, cos
        features, pos = self.backbone(samples)
        # mask is already interpolated on the feature maps. torch.Size([690, 2, 256]) ,  mask, torch.Size([2, 30, 23])
        src, mask = features[-1].decompose()
        assert mask is not None
        # default pipeline
        embedweight = self.refpoint_embed.weight  # object query
        # prepare for dn
        input_query_label, input_query_bbox, attn_mask, mask_dict = \
            prepare_for_dn(dn_args, embedweight, src.size(0), self.training, self.num_queries, self.num_classes,
                           self.hidden_dim, self.label_enc)
        # ------
        # if self.sgdt is not None:
        # dn_args= (targets, args.scalar, args.label_noise_scale, args.box_noise_scale, args.num_patterns)
        self.sgdt.src_key_padding_mask = mask  # B, H, W; to change to B, N use .flatten(1)
        bs, c, h, w = src.shape  # torch.Size([2, 256, 25, 32])
        self.sgdt.feat_map_size = (h, w)
        # self.sgdt.targets = targets

        if sgdt_args is not None and len(sgdt_args) > 0:
            targets = sgdt_args[0]

            self.sgdt.set_sgdt_targets(targets, feat_map_size=(h, w))  # , src_key_padding_mask=mask

            if self.training:
                gt_ratio_or_sigma = sgdt_args[1]
                self.sgdt.set_gt_ratio_or_sigma(gt_ratio_or_sigma)

            #  self.input_proj(src) only changes the number of channels, not the w, h: torch.Size([2, 2048, 28, 38])
            #  -> torch.Size([2, 256, 28, 38]), so we can safely use the h, w here, even later src goes
            #  through self.input_proj(src)

            # self.sgdt.set_sgdt_targets(targets, feat_map_size=(h, w)) # , src_key_padding_mask=mask
            # if teacher_encoder_output_list is not None:
            #     self.sgdt.teacher_encoder_output_list = teacher_encoder_output_list
        # pos[-1] input_query_bbox, object queries.
        hs, reference, sgdt_output_list, encoder_output_list = self.transformer(
            self.input_proj(src), mask, input_query_bbox, pos[-1],  # src, mask, refpoint_embed, pos_embed, tgt
            tgt=input_query_label,  # torch.Size([320, 2, 256]) decoder features, or decoder embedding,
            attn_mask=attn_mask,

            mask_dict=mask_dict,
            class_embed=self.class_embed,
            sgdt=self.sgdt,
            teacher_encoder_output_list=teacher_encoder_output_list,
            skip_teacher_model_decoder_forward=training_skip_forward_decoder,
            # sgdt_targets=sgdt_targets,
            # targets=targets,
            # proposal_processor=proposal_processor,  # proposal_processor=None,
            # input_img_sizes=input_img_sizes,
            # token_scoring_gt_generator=self.token_scoring_gt_generator,
        )

        sgdt_out = dict(
            sgdt_output_list=sgdt_output_list,
            # sgdt_targets=self.sgdt.sgdt_targets,
            # sgdt_target_raw=self.sgdt.sgdt_target_raw,
            encoder_output_list=encoder_output_list,
            teacher_encoder_output_list=teacher_encoder_output_list,
            src_key_padding_mask=self.sgdt.src_key_padding_mask,
            sgdt=self.sgdt
        )

        if len(self.sgdt.auxiliary_fg_bg_cls_encoder_layer_ids) > 0:
            token_ft_bg_predict_out = self.sgdt.token_fg_bg_classier(encoder_output_list)
            sgdt_out.update(token_ft_bg_predict_out)

        if training_skip_forward_decoder:
            return sgdt_out, mask_dict

        if self.training and self.train_token_scoring_only:
            return sgdt_out, mask_dict
        # ------------------------

        if not self.bbox_embed_diff_each_layer:
            reference_before_sigmoid = inverse_sigmoid(reference)
            tmp = self.bbox_embed(hs)
            tmp[..., :self.query_dim] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
        else:
            reference_before_sigmoid = inverse_sigmoid(reference)
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                tmp = self.bbox_embed[lvl](hs[lvl])
                tmp[..., :self.query_dim] += reference_before_sigmoid[lvl]
                outputs_coord = tmp.sigmoid()
                outputs_coords.append(outputs_coord)
            outputs_coord = torch.stack(outputs_coords)

        outputs_class = self.class_embed(hs)
        # dn post process
        outputs_class, outputs_coord = dn_post_process(outputs_class, outputs_coord, mask_dict)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        # ------------------ tti modification
        out.update(sgdt_out)
        return out, mask_dict


class SetCriterion(SetCriterionOld):

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses,
                 sgdt=None,
                 train_token_scoring_only=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss

        """

        super().__init__(num_classes, matcher, weight_dict, focal_alpha, losses)

        self.sgdt = sgdt
        self.train_token_scoring_only = train_token_scoring_only

        self.token_classification_loss = None
        if len(self.sgdt.auxiliary_fg_bg_cls_encoder_layer_ids) > 0:
            self.token_classification_loss = TokenFgBgClassificationLoss(sgdt=sgdt, weight_dict=weight_dict)


    def cal_decoder_distill_loss(self, outputs, targets, mask_dict=None, return_indices=False):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        # Compute all the requested losses
        losses = {}

        for loss in self.losses:
            losses.update(self._get_decoder_distill_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self._get_decoder_distill_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

    def _get_decoder_distill_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.decoder_distill_loss_labels,
            # 'cardinality': self.loss_cardinality,
            'boxes': self.decoder_distill_loss_boxes,
            # 'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def decoder_distill_loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses


    def decoder_distill_loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def forward(self, outputs, targets, mask_dict=None, return_indices=False, proposals=None):

        sgdt_losses = {}  # sgdt loss
        if not self.sgdt.disable_fg_scale_supervision and \
                'sgdt_output_list' in outputs and len(outputs['sgdt_output_list']) > 0:
            sgdt_output_list = outputs['sgdt_output_list']

            # sgdt_targets = outputs['sgdt_targets'] # TODO: remove this from outputs assignment
            # sgdt_target_raw = outputs['sgdt_target_raw']

            sgdt_targets = self.sgdt.sgdt_targets
            sgdt_target_raw = self.sgdt.sgdt_target_raw

            for i, sgdt_output in enumerate(sgdt_output_list):

                l_dict = self.sgdt.token_scoring_loss.cal_loss(
                    sgdt_output=sgdt_output, sgdt_targets=sgdt_targets)
                # l_dict = self.loss_fg_scale_score(sgdt_output, sgdt_targets)
                # l_dict = self.loss_fg_scale_score_ce(sgdt_output, sgdt_targets)

                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                sgdt_losses.update(l_dict)

                if self.sgdt.token_adaption_visualization:
                    vis_tool = VisualizeToken(targets=targets,
                                              sgdt_target_raw=sgdt_target_raw,
                                              sgdt_targets=sgdt_targets,
                                              sgdt_output=sgdt_output
                                              )
                    vis_tool.visualize_token_adaption(sub_dir=self.sgdt.visualization_out_sub_dir)
                    # vis_tool.save_intermediate_result()
                    # vis_tool.save_significant_score_and_split_tokens(sub_dir=self.visualization_out_sub_dir)
                # if self.save_intermediate_result:

        # calculate the feature loss during training

        if self.sgdt.feature_distiller_loss is not None:  # and self.training
            encoder_output_list = [x['feat'] for x in outputs['encoder_output_list']]

            if outputs['teacher_encoder_output_list'] is not None:
                teacher_encoder_output_list = [x['feat'] for x in outputs['teacher_encoder_output_list']]

                assert self.sgdt.distiller == 'separate_trained_model'
                assert len(teacher_encoder_output_list) > 0 and len(encoder_output_list) > 0
                # we always use the last encoder layer to conduct feature distillation
                teacher_transformer_feat = teacher_encoder_output_list[-1].clone().detach()
                student_transformer_feat = encoder_output_list[-1]
            else:
                # conduct distillation for the last single non-sgdt layers.
                normal_layer_ids, sgdt_layer_ids = split_encoder_layer(
                    encoder_layer_config=self.sgdt.encoder_layer_config)

                assert len(sgdt_layer_ids) > 0 and len(normal_layer_ids) > 0
                student_encoder_layer_id = normal_layer_ids[-1]
                teacher_encoder_layer_id = sgdt_layer_ids[-1]

                student_transformer_feat = encoder_output_list[student_encoder_layer_id]

                if self.sgdt.feature_distillation_teacher_feat_with_grad:
                    teacher_transformer_feat = encoder_output_list[teacher_encoder_layer_id]
                else:
                    teacher_transformer_feat = encoder_output_list[teacher_encoder_layer_id].clone().detach()

            # transform the transformer_feat to conv feat map
            student_feat = sgdt_token2feat(student_transformer_feat, feat_map_size=self.sgdt.feat_map_size,
                                           mask=self.sgdt.src_key_padding_mask.flatten(
                                               1))  # torch.Size([2, 256, 30, 32])
            teacher_feat = sgdt_token2feat(teacher_transformer_feat, feat_map_size=self.sgdt.feat_map_size,
                                           mask=self.sgdt.src_key_padding_mask.flatten(
                                               1))  # torch.Size([2, 256, 30, 32])

            # gt_bboxes = [unnormalize_box(box_normalized=t['boxes'],
            #                              input_img_size=t['size']).float() for t in self.sgdt.targets]
            # img_metas = [dict(img_shape=t['size']) for t in self.sgdt.targets]

            gt_bboxes = [unnormalize_box(box_normalized=t['boxes'],
                                         input_img_size=t['size']).float() for t in targets]
            img_metas = [dict(img_shape=t['size']) for t in targets]
            # features of the sgdt output feature with the features of the previous encoder layer.
            # tensor(0.0459, device='cuda:1', grad_fn=<AddBackward0>)
            sgdt_losses['sgdt_feature_distillation_loss'] = self.sgdt.feature_distiller_loss(
                preds_S=student_feat, preds_T=teacher_feat,  # Bs*C*H*W, student's feature map
                gt_bboxes=gt_bboxes, img_metas=img_metas)  #

        if self.sgdt.with_sgdt_attention_loss:  # and self.training
            assert self.sgdt.sgdt_attention_loss is not None

            encoder_attn_logit_map_list = [x['attn_map_logits'] for x in outputs['encoder_output_list']]

            if outputs['teacher_encoder_output_list'] is not None:
                # offline distillation

                teacher_encoder_attn_logit_map_list = [
                    x['attn_map_online_teacher'] for x in outputs['teacher_encoder_output_list']
                    if 'attn_map_online_teacher' in x and x['attn_map_online_teacher'] is not None]
                if len(teacher_encoder_attn_logit_map_list) == 0:
                    teacher_encoder_attn_logit_map_list = [
                        x['attn_map_logits'] for x in outputs['teacher_encoder_output_list']
                        if 'attn_map_logits' in x and x['attn_map_logits'] is not None]

                assert len(teacher_encoder_attn_logit_map_list) > 0 and len(encoder_attn_logit_map_list) > 0

                # we always use the last encoder layer to conduct feature distillation
                student_attn_logit_map = encoder_attn_logit_map_list[-1]
                teacher_attn_logit_map = teacher_encoder_attn_logit_map_list[-1]

            elif 'attn_map_online_teacher' in outputs['encoder_output_list'][-1]:
                # Online distillation
                teacher_attn_logit_map = outputs['encoder_output_list'][-1]['attn_map_online_teacher']
                student_attn_logit_map = encoder_attn_logit_map_list[-1]
                assert student_attn_logit_map.shape == teacher_attn_logit_map.shape and not \
                    (torch.equal(teacher_attn_logit_map, student_attn_logit_map))
                # bug check in case I passed attn of teacher to the student

            else:
                # conduct distillation for the last single non-sgdt layers.
                normal_layer_ids, sgdt_layer_ids = split_encoder_layer(
                    encoder_layer_config=self.sgdt.encoder_layer_config)

                assert len(sgdt_layer_ids) > 0 and len(normal_layer_ids) > 0
                student_encoder_layer_id = normal_layer_ids[-1]
                teacher_encoder_layer_id = sgdt_layer_ids[-1]

                student_attn_logit_map = encoder_attn_logit_map_list[student_encoder_layer_id]
                teacher_attn_logit_map = encoder_attn_logit_map_list[teacher_encoder_layer_id]

            # the order is important.
            # outputs['sgdt_output_list'][-1]['valid_tokens']
            if self.sgdt.attn_distillation_teacher_with_grad:
                sgdt_losses['sgdt_attention_distillation_loss'] = self.sgdt.sgdt_attention_loss(
                    input=student_attn_logit_map, target=teacher_attn_logit_map,
                    # examples always use .detach()
                    valid_tokens_float=self.sgdt.valid_tokens_float,
                    top_k=100 if self.sgdt.attention_loss_top_100_token else None
                    # TODO: change -1 to be a more general version.
                )
            else:
                sgdt_losses['sgdt_attention_distillation_loss'] = self.sgdt.sgdt_attention_loss(
                    input=student_attn_logit_map, target=teacher_attn_logit_map.detach(),
                    # examples always use .detach()
                    valid_tokens_float=self.sgdt.valid_tokens_float,
                    top_k=100 if self.sgdt.attention_loss_top_100_token else None
                    # TODO: change -1 to be a more general version.
                )
            # src_key_padding_mask

        if self.sgdt.args.with_decoder_prediction_distillation:  # and self.training
            # decoder_prediction
            encoder_attn_logit_map_list = [x['attn_map_logits'] for x in outputs['encoder_output_list']]

            if outputs['teacher_encoder_output_list'] is not None:
                # offline distillation

                teacher_encoder_attn_logit_map_list = [
                    x['attn_map_online_teacher'] for x in outputs['teacher_encoder_output_list']
                    if 'attn_map_online_teacher' in x and x['attn_map_online_teacher'] is not None]
                if len(teacher_encoder_attn_logit_map_list) == 0:
                    teacher_encoder_attn_logit_map_list = [
                        x['attn_map_logits'] for x in outputs['teacher_encoder_output_list']
                        if 'attn_map_logits' in x and x['attn_map_logits'] is not None]

                assert len(teacher_encoder_attn_logit_map_list) > 0 and len(encoder_attn_logit_map_list) > 0

                # we always use the last encoder layer to conduct feature distillation
                student_attn_logit_map = encoder_attn_logit_map_list[-1]
                teacher_attn_logit_map = teacher_encoder_attn_logit_map_list[-1]

            elif 'attn_map_online_teacher' in outputs['encoder_output_list'][-1]:
                # Online distillation
                teacher_attn_logit_map = outputs['encoder_output_list'][-1]['attn_map_online_teacher']
                student_attn_logit_map = encoder_attn_logit_map_list[-1]
                assert student_attn_logit_map.shape == teacher_attn_logit_map.shape and not \
                    (torch.equal(teacher_attn_logit_map, student_attn_logit_map))
                # bug check in case I passed attn of teacher to the student

            else:
                # conduct distillation for the last single non-sgdt layers.
                normal_layer_ids, sgdt_layer_ids = split_encoder_layer(
                    encoder_layer_config=self.sgdt.encoder_layer_config)

                assert len(sgdt_layer_ids) > 0 and len(normal_layer_ids) > 0
                student_encoder_layer_id = normal_layer_ids[-1]
                teacher_encoder_layer_id = sgdt_layer_ids[-1]

                student_attn_logit_map = encoder_attn_logit_map_list[student_encoder_layer_id]
                teacher_attn_logit_map = encoder_attn_logit_map_list[teacher_encoder_layer_id]

            # the order is important.
            # outputs['sgdt_output_list'][-1]['valid_tokens']
            if self.sgdt.attn_distillation_teacher_with_grad:
                sgdt_losses['sgdt_attention_distillation_loss'] = self.sgdt.sgdt_attention_loss(
                    input=student_attn_logit_map, target=teacher_attn_logit_map,
                    # examples always use .detach()
                    valid_tokens_float=self.sgdt.valid_tokens_float,
                    top_k=100 if self.sgdt.attention_loss_top_100_token else None
                    # TODO: change -1 to be a more general version.
                )
            else:
                sgdt_losses['sgdt_attention_distillation_loss'] = self.sgdt.sgdt_attention_loss(
                    input=student_attn_logit_map, target=teacher_attn_logit_map.detach(),
                    # examples always use .detach()
                    valid_tokens_float=self.sgdt.valid_tokens_float,
                    top_k=100 if self.sgdt.attention_loss_top_100_token else None
                    # TODO: change -1 to be a more general version.
                )
            # src_key_padding_mask

        if self.sgdt.with_sgdt_transformer_feature_distill_loss:  # and self.training
            assert self.sgdt.sgdt_transformer_feature_distill_loss is not None
            sgdt_losses['sgdt_transformer_feature_distillation_loss'] = 0

            if outputs['teacher_encoder_output_list'] is not None:
                # offline distillation
                # teacher_encoder_feat_list = [x['feat'] for x in outputs['teacher_encoder_output_list']]
                teacher_encoder_feat_list = [
                    x['output_online_teacher'] for x in outputs['teacher_encoder_output_list']
                    if 'output_online_teacher' in x and x['output_online_teacher'] is not None
                ]
                if len(teacher_encoder_feat_list) == 0:
                    teacher_encoder_feat_list = [
                        x['feat'] for x in outputs['teacher_encoder_output_list']
                        if 'feat' in x and x['feat'] is not None
                    ]
                student_encoder_feat_list = [
                    x['feat'] for x in outputs['encoder_output_list'] if 'feat' in x and x['feat'] is not None
                ]
                assert len(teacher_encoder_feat_list) > 0 and len(student_encoder_feat_list) > 0

                # we always use the last encoder layer to conduct feature distillation
                teacher_transformer_feat = teacher_encoder_feat_list[-1]
                student_transformer_feat = student_encoder_feat_list[-1]  # torch.Size([713, 2, 256])

                sgdt_losses['sgdt_transformer_feature_distillation_loss'] += \
                    self.sgdt.sgdt_transformer_feature_distill_loss(
                        input=student_transformer_feat, target=teacher_transformer_feat.detach(),
                        valid_tokens_float=self.sgdt.valid_tokens_float, )
            else:
                loss_exist = False
                for x in outputs['encoder_output_list']:
                    if 'feat' in x and 'output_online_teacher' in x:
                        teacher_transformer_feat = x['output_online_teacher']  # torch.Size([713, 2, 256])
                        student_transformer_feat = x['feat']  # torch.Size([713, 2, 256])

                        loss_exist = True
                        if not self.sgdt.feature_distillation_teacher_feat_with_grad:
                            teacher_transformer_feat = teacher_transformer_feat.clone().detach()

                        sgdt_losses['sgdt_transformer_feature_distillation_loss'] += \
                            self.sgdt.sgdt_transformer_feature_distill_loss(
                                input=student_transformer_feat, target=teacher_transformer_feat,
                                valid_tokens_float=self.sgdt.valid_tokens_float, )
                assert loss_exist

        if self.sgdt.attention_map_evaluation and not self.training:
            if self.sgdt.attention_map_evaluation == 'attn_map_online_teacher':
                encoder_attn_map_list = [F.softmax(x['attn_map_online_teacher'], dim=-1)
                                         for x in outputs['encoder_output_list']
                                         if 'attn_map_online_teacher' in x]
            else:
                encoder_attn_map_list = [F.softmax(x['attn_map_logits'], dim=-1)
                                         for x in outputs['encoder_output_list']
                                         if 'attn_map_logits' in x and x['attn_map_logits'] is not None]
            # conduct distillation for the last single non-sgdt layers. F.softmax(attn_output_weights, dim=-1)
            attn_map = encoder_attn_map_list[-1]
            N_head = attn_map.shape[0]
            sgdt_losses['attn_map'] = [attn_map[k] for k in range(N_head)]

            # sgdt_targets = self.sgdt.sgdt_targets
            # sgdt_target_raw = self.sgdt.sgdt_target_raw
            # N = self.sgdt.sgdt_targets['fg_gt'].shape[0]
            # fg_gt = sgdt_targets['fg_gt'].transpose(1, 0).bool()  # N, B  torch.Size([1064, 2])
            # fg_mask = fg_gt[:, None, :].repeat(1, N, 1)  # fg attn of all tokens
            # valid_token_float = (~(self.sgdt.src_key_padding_mask.flatten(1))).float()  # 2, N
            # num_valid_tokens = valid_token_float.sum(dim=1)
            # valid_attn = valid_token_float[:, :, None].repeat(1, 1, N)
            # for attn_map in encoder_attn_map_list:
            #     # student_encoder_layer_id = -2
            #     # teacher_encoder_layer_id = -1
            #     # normal_layer_ids, sgdt_layer_ids = split_encoder_layer(
            #     #     encoder_layer_config=self.sgdt.encoder_layer_config)
            #     #
            #     # assert len(sgdt_layer_ids) > 0 and len(normal_layer_ids) > 0
            #     # student_encoder_layer_id = normal_layer_ids[-1]
            #     # teacher_encoder_layer_id = sgdt_layer_ids[-1]
            #     # torch.Size([2, 1064, 1064])
            #     # student_attn_map = encoder_attention_map_list[student_encoder_layer_id]
            #     # teacher_attn_map = encoder_attention_map_list[teacher_encoder_layer_id]
            #     # N = attn_map.shape[-1]
            #
            #     encoder_layer_output_fg_attn = (attn_map * fg_mask * valid_attn).sum(dim=(1, 2)) / num_valid_tokens
            #     # teacher_fg_attn = (teacher_attn_map * fg_mask * valid_attn).sum(dim=(1, 2)) / num_valid_tokens
            #     fg_attn.append(encoder_layer_output_fg_attn)

            # save for each image, B, H, N, N -> H, N, N
            # N_head = fg_attn.shape[0]
            # sgdt_losses['fg_attn'] = [fg_attn[k] for k in range(N_head)]
            # sgdt_losses['fg_attn'] = [[x[0] for x in fg_attn], [x[1] for x in fg_attn]  for k in range(N_head])]
        # if self.training and self.train_token_scoring_only:
        #     return sgdt_losses
        # --------------

        if len(self.sgdt.auxiliary_fg_bg_cls_encoder_layer_ids) > 0:
            layer_ids = self.sgdt.auxiliary_fg_bg_cls_encoder_layer_ids
            token_fg_bg_cls_loss = self.token_classification_loss(outputs, sgdt=self.sgdt, layer_ids=layer_ids)
            sgdt_losses.update(token_fg_bg_cls_loss)

        if self.sgdt.training_only_distill_student_attn or self.sgdt.args.ignore_detr_loss:
            return sgdt_losses

        losses, indices_list = super().forward(outputs=outputs, targets=targets, mask_dict=mask_dict,
                                               return_indices=True)
        losses.update(sgdt_losses)

        if return_indices:
            return losses, indices_list
        return losses


def build_SGDT_DABDETR(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    sgdt = build_sgdt(args)

    # feature_distiller = build_feature_distiller(args)

    model = SGDT_DABDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,

        # to add new parameters
        sgdt=sgdt,
        # feature_distiller=feature_distiller,
        # train_token_scoring_only=args.train_token_scoring_only,
    )

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    # dn loss
    if args.use_dn:
        weight_dict['tgt_loss_ce'] = args.cls_loss_coef
        weight_dict['tgt_loss_bbox'] = args.bbox_loss_coef
        weight_dict['tgt_loss_giou'] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # # TODO this is a hack
    # if args.aux_loss:
    #     aux_weight_dict = {}
    #     for i in range(args.dec_layers - 1):
    #         aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    #     weight_dict.update(aux_weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers * 2 - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]

    # # ------------------------
    num_sgdt_layer = get_num_sgdt_layer(args.encoder_layer_config)
    if num_sgdt_layer > 0:
        # Define, but not necessarily use.
        sgdt_weight_dict = dict(
            sgdt_loss_fg=args.sgdt_loss_fg_coef,
            sgdt_loss_token_significance=args.sgdt_loss_token_significance_coef,
            sgdt_loss_small_scale=args.sgdt_loss_small_scale_coef,
        )
        for i in range(num_sgdt_layer):
            weight_dict.update({k + f'_{i}': v for k, v in sgdt_weight_dict.items()})
    weight_dict['sgdt_feature_distillation_loss'] = 1.0
    weight_dict['sgdt_attention_distillation_loss'] = args.sgdt_attention_loss_coef
    weight_dict['sgdt_transformer_feature_distillation_loss'] = args.sgdt_transformer_feature_distillation_loss_coef

    # parser.add_argument('--sgdt_decoder_pred_cls_loss_coef', default=2.0, type=float,)
    # parser.add_argument('--sgdt_decoder_pred_loc_loss_coef', default=5.0, type=float,)
    sgdt_weight_dict = {}
    for i in range(args.dec_layers * 2 - 1):
        sgdt_weight_dict.update(
            {
                'sgdt_decoder_pred_cls' + f'_{i}': args.sgdt_decoder_pred_cls_loss_coef,
                'sgdt_decoder_loc_cls' + f'_{i}': args.sgdt_decoder_pred_loc_loss_coef,
             }
        )
    weight_dict.update(sgdt_weight_dict)

    num_layer = len(sgdt.auxiliary_fg_bg_cls_encoder_layer_ids)
    if num_layer > 0:
        # Define, but not necessarily use.
        sgdt_weight_dict = dict(
            sgdt_loss_fg=1.0,  # args.sgdt_loss_fg_coef
            # sgdt_loss_token_significance=args.sgdt_loss_token_significance_coef,
            # sgdt_loss_small_scale=args.sgdt_loss_small_scale_coef,
        )
        for i in range(num_layer):
            weight_dict.update({k + f'_{i}': v for k, v in sgdt_weight_dict.items()})

    if args.training_only_distill_student_attn:
        weight_keys = list(weight_dict.keys()).copy()
        for w_k in weight_keys:
            if w_k.find('attention') == -1:
                del weight_dict[w_k]
                print(f'Disabling loss for {w_k}')

    if args.loss_disable_ignore_keywords:
        loss_disable_ignore_keywords = args.loss_disable_ignore_keywords if args.loss_disable_ignore_keywords else []
        if len(loss_disable_ignore_keywords) > 0:
            weight_keys = list(weight_dict.keys()).copy()
            for w_k in weight_keys:
                disable_ignore = False
                for keyword in loss_disable_ignore_keywords:
                    if w_k.find(keyword) > -1:
                        disable_ignore = True

                if not disable_ignore:
                    del weight_dict[w_k]
                    print(f'Disabling loss for {w_k}')

    # if args.train_token_scoring_only:
    #     weight_keys = list(weight_dict.keys()).copy()
    #     for w_k in weight_keys:
    #         if w_k.find('sgdt') == -1:
    #             del weight_dict[w_k]
    #             print(f'Disabling loss for {w_k}')
    # # ------------------------

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses,
                             # -------------
                             sgdt=sgdt,
                             )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
