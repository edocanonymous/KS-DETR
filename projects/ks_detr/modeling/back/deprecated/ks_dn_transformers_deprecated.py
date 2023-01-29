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

import torch
import torch.nn as nn

from detrex.layers import (
    FFN,
    MLP,
    # BaseTransformerLayer,
    ConditionalCrossAttention,
    ConditionalSelfAttention,
    # MultiheadAttention,
    # TransformerLayerSequence,
    get_sine_pos_embed,
)
from detrex.utils.misc import inverse_sigmoid

from projects.ks_detr.modeling.layers import (
    KSTransformerLayerSequence,
    KSBaseTransformerEncoderLayer,
    KSMultiheadAttentionWithGT,
)


from projects.ks_detr.modeling.ks_utils.smlp import sMLP, get_valid_token_mask, TokenScoringConfigParser
# from models.ksgt.scoring_gt import resize_ksgt_target
# from projects.ks_detr.modeling.DN_DAB_DETR.dn_components import dn_post_process
from projects.ks_detr.modeling.ks_utils.proposal import update_targets_with_proposals
from projects.ks_detr.modeling.ks_utils.ks_components import parser_encoder_decoder_layers
import numpy as np

EncoderLayerType = {
    'regular': dict(LayerType=KSBaseTransformerEncoderLayer, AttentionType=KSMultiheadAttentionWithGT) ,
    # 'WithGT': TransformerEncoderWithGTLayer,
    # 'WithGTFeature': TransformerEncoderWithGTFeatureLayer,
    #
    # 'DualAttnShareVOutProjFFN': TransformerEncoderDualAttnLayerShareVOutProjFFN,
    # # 'DualAttnShareVOutProjFFN': TransformerEncoderDualAttnLayerShareVOutProjFFN,
    # 'DualAttnShareVFFN': TransformerEncoderDualAttnLayerShareVFFN,
    # 'DualAttnShareV': TransformerEncoderDualAttnLayerShareV,
    # 'DualAttnShareAttnOutProjFFN': TransformerEncoderDualAttnLayerShareAttnOutProjFFN,
    # 'DualAttnShareAttnFFN': TransformerEncoderDualAttnLayerShareAttnFFN,
    # 'DualAttnShareAttn': TransformerEncoderDualAttnLayerShareAttn,
    #
    # 'TripleAttnShareOutProjFFN': TransformerEncoderTripleAttnLayerShareOutProjFFN,

    # 'regular': TransformerEncoderLayer,
    # 'ksgt': TransformerEncoderSGDTLayer,
    # 'ksgtv0': TransformerEncoderSGDTLayerV0NoMaskUpdate,
    # 'ksgtv1': TransformerEncoderSGDTLayerV1,
    # 'ksgt+k': TransformerEncoderSGDTLayerUpdateK,
    # 'ksgt+qk': TransformerEncoderSGDTLayerUpdateQK,
    # 'ksgt+v': TransformerEncoderSGDTLayerUpdateV,
    #
    # 'ksgt+qkv': TransformerEncoderSGDTLayerUpdateQKV,
    # 'ksgt+mha+out': TransformerEncoderSGDTLayerUpdateMHAOut,
    # 'ksgt+mha+feature': TransformerEncoderSGDTLayerUpdateMHAFeature,
    # 'ksgt+ffn+out': TransformerEncoderSGDTLayerUpdateFFNOut,
    # 'ksgt+ffn+feature': TransformerEncoderSGDTLayerUpdateFFNFeature,
    #
    # 'ksgtSharedAttn': TransformerEncoderSharedSelfAttnLayer,
    # 'ksgtv1FreezeSelfAttn': TransformerEncoderSGDTLayerUpdateKExtendedSelfAttn,
    # 'ksgtMarkFgBg': TransformerEncoderMarkFgBgLayer,
    # 'ksgtRandomMarkFgBg': TransformerEncoderRandomMarkFgBgLayer,
    #
    # # teacher student both in a single encoder layer
    # 'parallelSTECMarkFgBgVFreezeAll': TransformerEncoderLayerSTMarkFgBgShareQVFreezeAllExceptWK,
    # 'parallelSTECMarkFgBgShareVNoFreeze': TransformerEncoderLayerSTMarkFgBgShareVNoFreeze,
    # 'parallelSTECShareSGDTUpdateKAttn': TransformerEncoderLayerSTShareSGDTUpdateKAttn,
    # 'parallelSTECSGDTShareVNoFreeze': TransformerEncoderLayerSTSGDTShareVNoFreeze,
    # 'parallelSTECSGDTShareVOutProjFFN': TransformerEncoderLayerSTSGDTShareVOutProjFFN,
    # 'parallelSTECSGDTShareVFFN': TransformerEncoderLayerSTSGDTShareVFFN,
    # 'parallelSTECSGDTShareAtnOutProjFFN': TransformerEncoderLayerSTSGDTShareAttnOutProjFFN,
    #
    # 'parallelsMLPQKShareVOutProjFFN': TransformerEncoderLayersMLPQKShareVOutProjFFN,
    # 'parallelMarkFg1Bg0QKShareVOutProjFFN': TransformerEncoderLayerMarkFg1Bg0QKShareVOutProjFFN,
}


def generate_transformer_encoder_layers(EncoderLayerTypeDict, encoder_layer_config):
    encoder_layer_conf_list = parser_encoder_decoder_layers(encoder_layer_config)

    encoder_layer_list = []
    for l_type, num_l in encoder_layer_conf_list:
        assert l_type in EncoderLayerTypeDict and num_l > 0

        encoder_layer = EncoderLayerTypeDict[l_type](d_model, nhead, dim_feedforward,
                                                     dropout, activation, normalize_before,
                                                     )
        encoder_layer_list.append([encoder_layer, num_l])
    return encoder_layer_list



class KSDNDetrTransformerEncoder(KSTransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.0,
        activation: nn.Module = nn.PReLU(),
        # num_layers: int = None,
        post_norm: bool = False,
        batch_first: bool = False,

        encoder_layer_config: str = None,  # 'regular_6'
    ):




        super(KSDNDetrTransformerEncoder, self).__init__(
            transformer_layers=KSBaseTransformerEncoderLayer(
                attn=KSMultiheadAttentionWithGT(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    batch_first=batch_first,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                    activation=activation,
                ),
                norm=nn.LayerNorm(normalized_shape=embed_dim),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),

            # num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)

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

        intermediate_output_dict = {}
        for layer in self.layers:
            position_scales = self.query_scale(query)
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos * position_scales,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query, intermediate_output_dict


class KSDNDetrTransformerDecoder(KSTransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.0,
        activation: nn.Module = nn.PReLU(),
        num_layers: int = None,
        modulate_hw_attn: bool = True,
        post_norm: bool = True,
        return_intermediate: bool = True,
        batch_first: bool = False,
    ):
        super(KSDNDetrTransformerDecoder, self).__init__(
            transformer_layers=KSBaseTransformerEncoderLayer(
                attn=[
                    ConditionalSelfAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=batch_first,
                    ),
                    ConditionalCrossAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=batch_first,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                    activation=activation,
                ),
                norm=nn.LayerNorm(
                    normalized_shape=embed_dim,
                ),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim

        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)

        self.bbox_embed = None
        if modulate_hw_attn:
            self.ref_anchor_head = MLP(self.embed_dim, self.embed_dim, 2, 2)
        self.modulate_hw_attn = modulate_hw_attn

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

        for idx in range(num_layers - 1):
            self.layers[idx + 1].attentions[1].query_pos_proj = None

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
        anchor_box_embed=None,

        intermediate_output_dict=None,
        **kwargs,
    ):
        intermediate = []

        reference_points = anchor_box_embed.sigmoid()
        refpoints = [reference_points]

        for idx, layer in enumerate(self.layers):
            obj_center = reference_points[..., : self.embed_dim]
            query_sine_embed = get_sine_pos_embed(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)

            # do not apply transform in position in the first decoder layer
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)

            # apply position transform
            query_sine_embed = query_sine_embed[..., : self.embed_dim] * position_transform

            if self.modulate_hw_attn:
                ref_hw_cond = self.ref_anchor_head(query).sigmoid()
                query_sine_embed[..., self.embed_dim // 2 :] *= (
                    ref_hw_cond[..., 0] / obj_center[..., 2]
                ).unsqueeze(-1)
                query_sine_embed[..., : self.embed_dim // 2] *= (
                    ref_hw_cond[..., 1] / obj_center[..., 3]
                ).unsqueeze(-1)

            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                is_first_layer=(idx == 0),
                **kwargs,
            )

            # iter update
            if self.bbox_embed is not None:
                temp = self.bbox_embed(query)
                temp[..., : self.embed_dim] += inverse_sigmoid(reference_points)
                new_reference_points = temp[..., : self.embed_dim].sigmoid()

                if idx != self.num_layers - 1:
                    refpoints.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query)
        # TODO: return intermediate_output_dict
        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(refpoints).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2),
                ]

        return query.unsqueeze(0)


class KSDNDetrTransformer(nn.Module):
    def __init__(self, encoder=None, decoder=None):
        super(KSDNDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, anchor_box_embed, pos_embed, target=None, attn_mask=None, **kwargs):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)

        mask = mask.view(bs, -1)
        # intermediate_output_dict save the intermediate output of each encoder, decoder layer
        # They can be used for distillation, attention sharing, etc..
        memory, intermediate_output_dict = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask,

            **kwargs,
        )
        # Update the intermediate_output_dict
        hidden_state, references, intermediate_output_dict = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            attn_masks=attn_mask,
            anchor_box_embed=anchor_box_embed,

            intermediate_output_dict=intermediate_output_dict,
            **kwargs,
        )

        return hidden_state, references, intermediate_output_dict

# class KSDNDetrDualAttnTransformer(KSDNDetrTransformer):
#
#     def forward(self, x, mask, anchor_box_embed, pos_embed, target=None, attn_mask=None, **kwargs):
#         bs, c, h, w = x.shape
#         x = x.view(bs, c, -1).permute(2, 0, 1)
#         pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
#
#         mask = mask.view(bs, -1)
#         # intermediate_output_dict save the intermediate output of each encoder, decoder layer
#         # They can be used for distillation, attention sharing, etc..
#         memory, *encoder_intermediate_output_list = self.encoder(
#             query=x,
#             key=None,
#             value=None,
#             query_pos=pos_embed,
#             query_key_padding_mask=mask,
#
#             **kwargs,
#         )
#         if encoder_intermediate_output_list:
#             encoder_intermediate_output_list = encoder_intermediate_output_list[0]
#
#         # Update the intermediate_output_dict
#         hidden_state, references, *decoder_intermediate_output_list = self.decoder(
#             query=target,
#             key=memory,
#             value=memory,
#             key_pos=pos_embed,
#             attn_masks=attn_mask,
#             anchor_box_embed=anchor_box_embed,
#
#             encoder_intermediate_output_list=encoder_intermediate_output_list,
#             **kwargs,
#         )
#         if decoder_intermediate_output_list:
#             decoder_intermediate_output_list = decoder_intermediate_output_list[0]
#
#         intermediate_output_dict = dict(
#             encoder_intermediate_output_list=encoder_intermediate_output_list,
#             decoder_intermediate_output_list=decoder_intermediate_output_list
#         )
#
#         if encoder_intermediate_output_list:  # and self.training)
#             count = 0
#             for encoder_out in encoder_intermediate_output_list:
#                 if 'feat_t' in encoder_out:
#                     teacher_memory = encoder_out['feat_t']
#                     count += 1
#
#                     hs_t, references_t, *decoder_intermediate_output_list_t = self.decoder(
#                         query=target,
#                         key=teacher_memory,
#                         value=teacher_memory,
#                         key_pos=pos_embed,
#                         attn_masks=attn_mask,
#                         anchor_box_embed=anchor_box_embed,
#
#                         encoder_intermediate_output_list=encoder_intermediate_output_list,
#                         **kwargs,
#                     )
#
#                     # # self.bbox_embed may be per level (self.bbox_embed[lvl](hs[lvl])), so I cannot put everything
#                     # # into a single tensor. So the following line is deprecated.
#                     # # hs, references = torch.cat([hs, hs_t], dim=0), torch.cat([references, references_t], dim=0)
#                     # TODO: return list,
#                     # hidden_state, references = [hidden_state, hs_t], [references, references_t]
#                     hidden_state, references = torch.cat([hidden_state, hs_t], dim=0), \
#                                                torch.cat([references, references_t], dim=0)
#
#                     if decoder_intermediate_output_list_t:
#                         decoder_intermediate_output_list_t = decoder_intermediate_output_list_t[0]
#                         intermediate_output_dict[
#                             'decoder_intermediate_output_list_t'] = decoder_intermediate_output_list_t
#
#             assert count <= 1, 'KSDNDetrDualAttnTransformer currently only support one encoder layer to have dual attn'
#
#         return hidden_state, references, intermediate_output_dict
#