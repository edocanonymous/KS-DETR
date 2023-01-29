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

import math
import torch
import torch.nn as nn

from detrex.layers import (
    # FFN,
    # BaseTransformerLayer,
    # MultiheadAttention,
    MultiScaleDeformableAttention,
    # TransformerLayerSequence,
)
from detrex.utils import inverse_sigmoid

# ----------------------
from projects.ks_detr.modeling.layers import (
    KSTransformerLayerSequence,
    DeformableEncoderLayerDict,
    DeformableDecoderLayerDict,
    KSBaseMultiScaleDeformableAttention,
)
from projects.deformable_detr.modeling import DeformableDetrTransformer
from .detr_module_utils import (
    concat_student_teacher_decoder_output,
    extend_student_teacher_decoder_output,
    extract_optional_output,
)
from .layers.transformer import generate_deformable_transformer_encoder_layers, \
    generate_deformable_transformer_decoder_layers
# ----------------------


class KSDeformableDetrTransformerEncoder(KSTransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        # num_layers: int = 6,
        post_norm: bool = False,
        num_feature_levels: int = 4,

        encoder_layer_config: str = None,  # 'regular_6'
    ):
        # =====================================-
        encoder_layer_list = generate_deformable_transformer_encoder_layers(
            encoder_layer_dict=DeformableEncoderLayerDict,
            encoder_layer_config=encoder_layer_config,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            feedforward_dim=feedforward_dim,
            ffn_dropout=ffn_dropout,
            num_feature_levels=num_feature_levels,
        )
        super(KSDeformableDetrTransformerEncoder, self).__init__(
            encoder_decoder_layer_list=encoder_layer_list
        )
        # =====================================-

        # super(KSDeformableDetrTransformerEncoder, self).__init__(
        #     transformer_layers=BaseTransformerLayer(
        #         attn=MultiScaleDeformableAttention(
        #             embed_dim=embed_dim,
        #             num_heads=num_heads,
        #             dropout=attn_dropout,
        #             batch_first=True,
        #             num_levels=num_feature_levels,
        #         ),
        #         ffn=FFN(
        #             embed_dim=embed_dim,
        #             feedforward_dim=feedforward_dim,
        #             output_dim=embed_dim,
        #             num_fcs=2,
        #             ffn_drop=ffn_dropout,
        #         ),
        #         norm=nn.LayerNorm(embed_dim),
        #         operation_order=("self_attn", "norm", "ffn", "norm"),
        #     ),
        #     num_layers=num_layers,
        # )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        encoder_intermediate_output_list = []
        for layer in self.layers:
            query, *intermediate_output = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

            encoder_intermediate_output_list.append(extract_optional_output(intermediate_output))
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query, encoder_intermediate_output_list


class KSDeformableDetrTransformerDecoder(KSTransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        # num_layers: int = 6,
        return_intermediate: bool = True,
        num_feature_levels: int = 4,
        decoder_layer_config: str = None,
    ):
        # --------------------
        decoder_layer_list = generate_deformable_transformer_decoder_layers(
            decoder_layer_dict=DeformableDecoderLayerDict,
            decoder_layer_config=decoder_layer_config,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            feedforward_dim=feedforward_dim,
            ffn_dropout=ffn_dropout,
            num_feature_levels=num_feature_levels,
        )
        super(KSDeformableDetrTransformerDecoder, self).__init__(
            encoder_decoder_layer_list=decoder_layer_list
        )
        num_layers = self.num_layers
        # --------------------

        # super(KSDeformableDetrTransformerDecoder, self).__init__(
        #     transformer_layers=BaseTransformerLayer(
        #         attn=[
        #             MultiheadAttention(
        #                 embed_dim=embed_dim,
        #                 num_heads=num_heads,
        #                 attn_drop=attn_dropout,
        #                 batch_first=True,
        #             ),
        #             MultiScaleDeformableAttention(
        #                 embed_dim=embed_dim,
        #                 num_heads=num_heads,
        #                 dropout=attn_dropout,
        #                 batch_first=True,
        #                 num_levels=num_feature_levels,
        #             ),
        #         ],
        #         ffn=FFN(
        #             embed_dim=embed_dim,
        #             feedforward_dim=feedforward_dim,
        #             output_dim=embed_dim,
        #             ffn_drop=ffn_dropout,
        #         ),
        #         norm=nn.LayerNorm(embed_dim),
        #         operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        #     ),
        #     num_layers=num_layers,
        # )
        self.return_intermediate = return_intermediate

        self.bbox_embed = None
        self.class_embed = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        reference_points=None,
        valid_ratios=None,
        **kwargs,
    ):
        decoder_intermediate_out_list = []

        output = query

        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                **kwargs,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), decoder_intermediate_out_list

        return output, reference_points, decoder_intermediate_out_list


class KSDeformableDetrTransformer(DeformableDetrTransformer):
    """Transformer module for Deformable DETR

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 300.
            Only used when as_two_stage is True.
    """

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, (MultiScaleDeformableAttention, KSBaseMultiScaleDeformableAttention)):
                m.init_weights()
        if not self.as_two_stage:
            nn.init.xavier_normal_(self.reference_points.weight.data, gain=1.0)
            nn.init.constant_(self.reference_points.bias.data, 0.0)
        nn.init.normal_(self.level_embeds)

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        query_embed,
        **kwargs,
    ):
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):  # torch.Size([2, 256, 10, 13]), torch.Size([2, 256, 80, 102])
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # torch.Size([2, 80, 102]) # b, h, w -> b, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            # Note: self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)

        # torch.Size([2, 10850, 256])
        feat_flatten = torch.cat(feat_flatten, 1)  # cat alone the spatial dimension, so it seems there is only one level feat
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        # =============================
        if kwargs.get('ksgt', None):  kwargs['ksgt'].mask = mask_flatten  # Set the mask
        # =============================

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        ) # tensor([[ 80, 102], [ 40,  51], [ 20,  26], [ 10,  13]], device='cuda:0')
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )  # tensor([  0,  8160, 10200, 10720], device='cuda:0')
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)
        # torch.Size([2, 4, 2])  # b, 4level, 2 dim for h, w
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )  # torch.Size([2, 10850, 4, 2])

        memory, *encoder_intermediate_output_list = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )
        # -------------
        encoder_intermediate_output_list = extract_optional_output(encoder_intermediate_output_list)
        # -------------
        bs, _, c = memory.shape
        if self.as_two_stage:  # not go inside this branch
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )

            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            )

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact))
            )
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points

        # decoder
        inter_states, inter_references, *decoder_intermediate_output_list = self.decoder(
            query=query,  # bs, num_queries, embed_dims
            key=None,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=query_pos,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 4
            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,  # bs, nlvl, 2
            **kwargs,
        )
        # ----
        decoder_intermediate_output_list = extract_optional_output(decoder_intermediate_output_list)
        intermediate_output_dict = dict(
            encoder_intermediate_output_list=encoder_intermediate_output_list,
            decoder_intermediate_output_list=decoder_intermediate_output_list,
            # target=target,  # Only for this method
            # pos_embed=pos_embed,  # The shape has changed
            # query_embed=query_embed,  # The shape has changed
        )
        # ------
        inter_references_out = inter_references
        if self.as_two_stage:
            return (
                inter_states,
                init_reference_out,
                inter_references_out,
                enc_outputs_class,
                enc_outputs_coord_unact,
            ) + (intermediate_output_dict,)
        return inter_states, init_reference_out, inter_references_out, None, None, intermediate_output_dict


class KSDeformableDetrMultiAttnTransformer(KSDeformableDetrTransformer):

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        query_embed,
        **kwargs,
    ):
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        # ============================-
        if kwargs.get('ksgt', None):  kwargs['ksgt'].mask = mask_flatten  # Set the mask
        # ============================-

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )

        memory, *encoder_intermediate_output_list = self.encoder(
            query=feat_flatten,  # torch.Size([2, 16500, 256])
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )
        encoder_intermediate_output_list = extract_optional_output(encoder_intermediate_output_list)

        bs, _, c = memory.shape
        if self.as_two_stage:  # not go inside this branch
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )

            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            )

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact))
            )
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points

        # decoder
        inter_states, inter_references, *decoder_intermediate_output_list = self.decoder(
            query=query,  # bs, num_queries, embed_dims
            key=None,  # bs, num_tokens, embed_dims  # === no key is used
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=query_pos,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 4
            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,  # bs, nlvl, 2
            **kwargs,
        )
        decoder_intermediate_output_list = extract_optional_output(decoder_intermediate_output_list)
        intermediate_output_dict = dict(
            encoder_intermediate_output_list=encoder_intermediate_output_list,
            decoder_intermediate_output_list=decoder_intermediate_output_list,

            # target=target,  # Only for this method
            # pos_embed=pos_embed,  # The shape has changed
            # query_embed=query_embed,  # The shape has changed
        )

        # inter_references_out = inter_references
        # if self.as_two_stage:
        #     return (
        #         inter_states,
        #         init_reference_out,
        #         inter_references_out,
        #         enc_outputs_class,
        #         enc_outputs_coord_unact,
        #     ) + (intermediate_output_dict,)


        # =================
        # the above are directly copied from its super class, without any modification.

        if encoder_intermediate_output_list:
            count = 0
            for encoder_out in encoder_intermediate_output_list:
                if 'feat_t' in encoder_out:
                    teacher_memories = encoder_out['feat_t']

                    # Dual attention does not output a list, and we need to transfer it to a list.
                    if not isinstance(teacher_memories, (list, tuple)):
                        teacher_memories = [teacher_memories]

                    count += 1
                    for teacher_memory in teacher_memories:
                        # ----------------------
                        # decoder
                        inter_states_t, inter_references_t, *decoder_intermediate_output_list_t = self.decoder(
                            query=query,  # bs, num_queries, embed_dims
                            key=None,  # bs, num_tokens, embed_dims
                            value=teacher_memory,  # bs, num_tokens, embed_dims ---
                            query_pos=query_pos,
                            key_padding_mask=mask_flatten,  # bs, num_tokens
                            reference_points=reference_points,  # num_queries, 4
                            spatial_shapes=spatial_shapes,  # nlvl, 2
                            level_start_index=level_start_index,  # nlvl
                            valid_ratios=valid_ratios,  # bs, nlvl, 2
                            **kwargs,
                        )
                        # ---------------------- The following sections require careful attention.
                        inter_states, inter_references = extend_student_teacher_decoder_output(
                            hs=inter_states, references=inter_references,
                            hs_t=inter_states_t, references_t=inter_references_t,
                        )

                        # if isinstance(self.decoder.bbox_embed, nn.ModuleList):  # self.decoder.bbox_embed = None
                            # Always return the intermediate out to keep the layer structure, because
                            # self.class_embed, self.bbox_embed = nn.ModuleList, nn.ModuleList, not a single
                            # MLP.
                            # ksgt.teacher_attn_return_no_intermediate_out is handle in detr, not in
                            # the Transformer.
                            # inter_states, inter_references = extend_student_teacher_decoder_output(
                            #     hs=inter_states, references=inter_references,
                            #     references_t=inter_references_t, hs_t=inter_states_t,
                            # )
                        # else:
                        #     ksgt = kwargs.get('ksgt', None)
                        #     # hidden_state: torch.Size([6, 2, 300, 256]), reference, torch.Size([2, 300, 2])
                        #     inter_states, inter_references = concat_student_teacher_decoder_output(
                        #         hs=inter_states, references=inter_references,
                        #         references_t=inter_references_t, hs_t=inter_states_t,
                        #         teacher_attn_return_no_intermediate_out=True if
                        #         ksgt and ksgt.teacher_attn_return_no_intermediate_out else False
                        #     )

                        if decoder_intermediate_output_list_t:
                            decoder_intermediate_output_list_t = decoder_intermediate_output_list_t[0]
                            intermediate_output_dict[
                                'decoder_intermediate_output_list_t'] = decoder_intermediate_output_list_t

            assert 0 < count, 'MultiAttnTransformer requires teacher memory to work '

        inter_references_out = inter_references
        if self.as_two_stage:  # TODO: update the code for two_stage
            return (
                       inter_states,
                       init_reference_out,
                       inter_references_out,
                       enc_outputs_class,
                       enc_outputs_coord_unact,
                   ) + (intermediate_output_dict,)

        return inter_states, init_reference_out, inter_references_out, None, None, intermediate_output_dict

