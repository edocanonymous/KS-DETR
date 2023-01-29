# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------


import math
import copy
import os
from typing import Optional, List
from util.misc import inverse_sigmoid

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.DN_DAB_DETR.attention import MultiheadAttention

# ------------------- TTI Modification
from models.DN_DAB_DETR.transformer import MLP, TransformerDecoder, TransformerEncoderLayer, \
    TransformerDecoderLayer,  _get_activation_fn, _get_clones,  gen_sineembed_for_position

from models.sgdt.sgdt_module import SGDT_module, get_valid_token_mask, TokenScoringConfigParser
# from models.sgdt.scoring_gt import resize_sgdt_target
from models.DN_DAB_DETR.dn_components import dn_post_process
from models.sgdt.scoring_gt import update_targets_with_proposals
from models.DN_DAB_DETR.transformer import Transformer as TransformerOld


def extract_adapted_token_pos_embed(adapted_token_dict, pos: Optional[Tensor]):
    """
    return extracted pos based on tokens_small_obj, tokens_to_discard in adapted_token_dict
    Args:
        adapted_token_dict: a dict included tokens_small_obj, tokens_to_discard
        pos:  position_embedding, (N, B, C), e.g., torch.Size([800, 2, 256])
            requires_grad = False (as sine encoding is used, not learnable position_embedding)
    Returns:
    """
    if pos is None:
        return pos
    else:
        assert not pos.requires_grad, 'If use learnable position_embedding, the code in extract_adapted_token_pos_embed' \
                                      'should be adapted to be differentiable.'

        # In case that we increase the number of tokens in sgdt module.
        if 'increase_resolution' in adapted_token_dict and adapted_token_dict['increase_resolution']:
            N_old, B, C = pos.shape
            N_new = adapted_token_dict['tokens_small_obj'].shape[0]
            # padding 0s is OK, even for padded tokens that are not used, because the mask will make sure they
            # will not participate the in later computation.
            padded_pos = pos.new_tensor(torch.zeros((N_new - N_old, B, C))).to(pos.device)
            pos = torch.cat([pos, padded_pos], dim=0)

        # We allow no token removing or splitting for ablation study.
        if adapted_token_dict['tokens_small_obj'].sum() == 0 or \
                adapted_token_dict['tokens_to_discard'].sum() == 0:
            return pos

        N, B, C = pos.shape
        adapted_pos = pos.clone()  # (N, B, C), e.g., torch.Size([800, 2, 256])  # pos: requires_grad = True grad

        if 'split_tok_indicators' in adapted_token_dict:  # process each image separately
            split_tok_indicators = adapted_token_dict['split_tok_indicators']  # list
            assert B == len(split_tok_indicators)
            for k in range(B):

                indicator = split_tok_indicators[k]

                if indicator.shape[1] == 0:  # B, N, C  (B is always 1)
                    continue

                pos_k = pos[:, k, :].unsqueeze(0)  # N, C -> 1, N, C
                # indicator = einops.rearrange(indicator, "b k d -> b d k")  #
                # indicators = rearrange(indicators, "b d k -> b k d")
                # patches = torch.einsum("b k d, b d c -> b k c", indicator, x_k)  # b, k, c, (1, top_k, c)
                patches = torch.einsum("b k d, b d c -> b k c", indicator, pos_k)  # b, k, c, (1, top_k, c)

                # The following two lines have the same operation with sgdt_module.py top_k for processing x.
                img_discard_token_ids = adapted_token_dict['tokens_to_discard'][:, k].bool()
                # x_small[img_discard_token_ids, k, :] = tokens_small_obj_new
                adapted_pos[img_discard_token_ids, k, :] = patches.squeeze(0)  # top_k, c

                patches_inverse = torch.einsum("b d k, b k c -> b d c",
                                               indicator.permute(0, 2, 1),  # b k d -> b d k
                                               patches  # b, k, c
                                               )
                adapted_pos[:, k, :] += patches_inverse.squeeze(0)  # 1, d, c -> d, c (d = N)

        else:
            # number to discard = number of small (for each image)
            # N, B, bool(), 1, small -> N, B, C
            tokens_small_obj = adapted_token_dict['tokens_small_obj'].unsqueeze(-1).repeat(1, 1, C)
            # N, B; bool(), 1, to discard -> N, B, C
            tokens_to_discard = adapted_token_dict['tokens_to_discard'].unsqueeze(-1).repeat(1, 1, C)
            # TODO: directly modify the value, differentiable?
            adapted_pos[tokens_to_discard] = pos[tokens_small_obj]  # check

        return adapted_pos


# class Transformer(TransformerOld):
#     def __init__(self, d_model=512, nhead=8, num_queries=300,
#                  num_encoder_layers=6,
#                  num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", normalize_before=False, return_intermediate_dec=False, query_dim=4,
#                  keep_query_pos=False, query_scale_type='cond_elewise', num_patterns=0, modulate_hw_attn=True,
#                  bbox_embed_diff_each_layer=False,
#
#                  # encoder_without_pos=False,
#                  encoder_layer_config='regular_6',
#                  train_token_scoring_only=False
#                  ):
#
#         # super().__init__(d_model=d_model, nhead=nhead, num_queries=num_queries, num_encoder_layers=num_encoder_layers,
#         #                  num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout,
#         #                  activation=activation, normalize_before=normalize_before,
#         #                  return_intermediate_dec=return_intermediate_dec, query_dim, keep_query_pos, query_scale_type,
#         #                  num_patterns, modulate_hw_attn, bbox_embed_diff_each_layer)
#
#         del self.encoder
#
#         # self.encoder_without_pos = encoder_without_pos
#
#         # # self.reclaim_padded_region = reclaim_padded_region
#         assert encoder_layer_config is not None and isinstance(encoder_layer_config, str)
#         if len(encoder_layer_config) == 0:
#             self.encoder = TransformerEmptyEncoder()
#         else:
#             # 'regular_6',  'regular_4-sgdtv1_1-sgdt_1'
#             layer_conf_split = encoder_layer_config.split('-')
#             encoder_layer_list = []
#             for l_conf in layer_conf_split:
#                 l_type_and_num = l_conf.split('_')
#                 assert len(l_type_and_num) == 2, f'The format of encoder layer config is wrong, ' \
#                                                  'expected length 2, e.g., regular_6, but got' \
#                                                  '{l_conf}'
#                 l_type, num_l = l_type_and_num[0], int(l_type_and_num[1])
#                 assert num_l > 0
#
#                 if l_type == 'regular':
#                     encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
#                                                             dropout, activation, normalize_before,
#                                                             )
#                 elif l_type == 'sgdt':
#
#                     encoder_layer = TransformerEncoderSGDTLayer(
#                         d_model, nhead, dim_feedforward,
#                         dropout, activation, normalize_before,
#                         # reclaim_padded_region=reclaim_padded_region,
#                         token_scoring_discard_split_criterion=token_scoring_discard_split_criterion,
#                     )
#                 elif l_type == 'sgdtv1':
#                     encoder_layer = TransformerEncoderSGDTLayerV1(
#                         d_model, nhead, dim_feedforward,
#                         dropout, activation, normalize_before,
#                         # ----------------
#                         token_scoring_discard_split_criterion=token_scoring_discard_split_criterion,
#                     )
#                 else:
#                     raise NotImplementedError(f'Encoder layer type {l_type} not implemented.')
#
#                 encoder_layer_list.append([encoder_layer, num_l])
#
#             encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
#             self.encoder = TransformerEncoder(encoder_layer_list, encoder_norm)
#
#         self._reset_parameters()
#
#         # ------------
#         # assert token_scoring_gt_generator is not None
#         # self.sgdt.token_scoring_gt_generator  = token_scoring_gt_generator
#         self.train_token_scoring_only = train_token_scoring_only
#
#
#     def forward(self, src, mask, refpoint_embed, pos_embed, tgt, attn_mask=None,
#                 sgdt_targets=None,  # for debugging only
#                 sigma=None,
#                 mask_dict=None,  # mask_dict=mask_dict,
#                 proposal_processor=None,
#                 class_embed=None,   # class_embed=class_embed
#                 input_img_sizes=None,  # input_img_sizes=input_img_sizes,
#                 targets=None,  # targets=targets,
#                 token_scoring_gt_generator=None,
#                 ):
#         """input:
#             mask: not None,
#             attn_mask: None,
#         Returns:
#
#         """
#         # flatten NxCxHxW to HWxNxC # h= 25, w = 32
#         bs, c, h, w = src.shape  # torch.Size([2, 256, 25, 32])
#         src = src.flatten(2).permute(2, 0, 1)
#         pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # torch.Size([2, 256, 25, 32]) -> torch.Size([800, 2, 256])
#         # refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)
#         # -------------------------
#
#         # sgdt_targets = self.sgdt.token_scoring_gt_generator .resize_sig_value_gt(
#         #     sgdt_targets, feat_map_size=(h, w))
#         # sgdt_targets = self.token_scoring_gt_generator(
#         #     sgdt_target_raw, feat_map_size=(h, w), feat_map_mask=mask)
#         # mask, (B, H, W)  -> (B, N), e.g., torch.Size([2, 30, 23]) -> torch.Size([2, 690])
#         mask = mask.flatten(1)  # torch.Size([2, 21, 30]) -> torch.Size([2, 630]), src = torch.Size([630, 2, 256])
#
#         # do not accept the adapted pos_embed, mask from encoder as zeros pos_embed might be fed
#         # mask is used as src_key_padding_mask not mask even encoder has 'mask' input.
#         memory, sgdt_output_list, pos_embed, mask = self.encoder(
#             src, src_key_padding_mask=mask,
#             pos=pos_embed if not self.encoder_without_pos else torch.zeros_like(pos_embed),
#             sgdt_targets=sgdt_targets,
#             feat_map_size=(h, w),
#             sigma=sigma
#         )
#
#         # if training token score only, we can skip the forward propagation of decoder.
#         if self.training and self.train_token_scoring_only:
#             return None, None, sgdt_output_list
#
#         if self.num_patterns > 0:
#             l = tgt.shape[0]
#             tgt[l - self.num_queries * self.num_patterns:] += \
#                 self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1)
#
#         hs, references = self.decoder(tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
#                                       pos=pos_embed, refpoints_unsigmoid=refpoint_embed)
#         return hs, references, sgdt_output_list
#

class TransformerSGDT(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_queries=300,
                 # num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,
                 # ----------------------
                 # num_encoder_sgdt_layers=0,
                 # encoder_sgdt_layer_version=0,
                 encoder_without_pos=False,
                 # reclaim_padded_region=False,
                 token_scoring_discard_split_criterion=None,
                 encoder_layer_config='regular_6',
                 # token_scoring_gt_generator=None,
                 ):

        super().__init__()

        self.encoder_without_pos = encoder_without_pos

        # # self.reclaim_padded_region = reclaim_padded_region
        assert encoder_layer_config is not None and isinstance(encoder_layer_config, str)
        if len(encoder_layer_config) == 0:
            self.encoder = TransformerEmptyEncoder()
        else:
            # 'regular_6',  'regular_4-sgdtv1_1-sgdt_1'
            layer_conf_split = encoder_layer_config.split('-')
            encoder_layer_list = []
            for l_conf in layer_conf_split:
                l_type_and_num = l_conf.split('_')
                assert len(l_type_and_num) == 2, f'The format of encoder layer config is wrong, ' \
                                                 'expected length 2, e.g., regular_6, but got' \
                                                 '{l_conf}'
                l_type, num_l = l_type_and_num[0], int(l_type_and_num[1])
                assert num_l > 0

                if l_type == 'regular':
                    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                            dropout, activation, normalize_before,
                                                            )
                elif l_type == 'sgdt':

                    encoder_layer = TransformerEncoderSGDTLayer(
                        d_model, nhead, dim_feedforward,
                        dropout, activation, normalize_before,
                        # reclaim_padded_region=reclaim_padded_region,
                        token_scoring_discard_split_criterion=token_scoring_discard_split_criterion,
                    )
                elif l_type == 'sgdtv1':
                    encoder_layer = TransformerEncoderSGDTLayerV1(
                        d_model, nhead, dim_feedforward,
                        dropout, activation, normalize_before,
                        # ----------------
                        token_scoring_discard_split_criterion=token_scoring_discard_split_criterion,
                    )
                else:
                    raise NotImplementedError(f'Encoder layer type {l_type} not implemented.')

                encoder_layer_list.append([encoder_layer, num_l])

            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            # --------------------------------------------
            # self.encoder = TransformerEncoder(encoder_layer_list, encoder_norm)
            self.encoder = TransformerSGDTEncoder(encoder_layer_list, encoder_norm)
            # --------------------------------------------

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        decoder_norm = nn.LayerNorm(d_model)

        # =======================================================
        # self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                   return_intermediate=return_intermediate_dec,
        #                                   d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos,
        #                                   query_scale_type=query_scale_type,
        #                                   modulate_hw_attn=modulate_hw_attn,
        #                                   bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        self.decoder = TransformerSGDTDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec,
                                              d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos,
                                              query_scale_type=query_scale_type,
                                              modulate_hw_attn=modulate_hw_attn,
                                              bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)
        # =======================================================
        self._reset_parameters()
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, d_model)

        # ====================
        # assert token_scoring_gt_generator is not None
        # self.token_scoring_gt_generator = token_scoring_gt_generator

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, refpoint_embed, pos_embed, tgt, attn_mask=None,
                sgdt_targets=None,  # for debugging only
                # token_scoring_gt_generator=None,
                sigma=None,
                mask_dict=None,  # mask_dict=mask_dict,
                proposal_processor=None,
                class_embed=None,   # class_embed=class_embed
                input_img_sizes=None,  # input_img_sizes=input_img_sizes,
                targets=None,  # targets=targets,
                token_scoring_gt_generator=None,
                ):
        # flatten NxCxHxW to HWxNxC # h= 25, w = 32
        bs, c, h, w = src.shape  # torch.Size([2, 256, 25, 32])
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # torch.Size([2, 256, 25, 32]) -> torch.Size([800, 2, 256])
        # refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)
        # -------------------------
        # sgdt_targets = resize_sgdt_target(
        #     sgdt_target_raw, feat_map_size=(h, w), feat_map_mask=mask)
        # sgdt_targets = self.token_scoring_gt_generator.resize_sig_value_gt(
        #     sgdt_targets, feat_map_size=(h, w))

        # mask, (B, H, W)  -> (B, N), e.g., torch.Size([2, 30, 23]) -> torch.Size([2, 690])
        mask = mask.flatten(1)  # torch.Size([2, 21, 30]) -> torch.Size([2, 630]), src = torch.Size([630, 2, 256])

        # do not accept the adapted pos_embed, mask from encoder as zeros pos_embed might be fed
        memory, sgdt_output_list, pos_embed, mask = self.encoder(
            src, src_key_padding_mask=mask,
            pos=pos_embed if not self.encoder_without_pos else torch.zeros_like(pos_embed),
            sgdt_targets=sgdt_targets,
            feat_map_size=(h, w),
            forward_layers='first_half',  # 'all', 'first_half',  'second_half'
            sigma=sigma,
        )

        if self.num_patterns > 0:
            l = tgt.shape[0]
            tgt[l - self.num_queries * self.num_patterns:] += \
                self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1)

        # hs, references = self.decoder(tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
        #                               pos=pos_embed, refpoints_unsigmoid=refpoint_embed)

        hs, references, sgdt_output_list_decoder = self.decoder(
            tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
            pos=pos_embed, refpoints_unsigmoid=refpoint_embed,

            sgdt_targets=sgdt_targets,  # only for debugging
            feat_map_size=(h, w),
            encoder=self.encoder,
            encoder_without_pos=self.encoder_without_pos,
            sigma=sigma,

            mask_dict=mask_dict,
            proposal_processor=proposal_processor,  # proposal_processor=None,
            class_embed=class_embed,
            input_img_sizes=input_img_sizes,
            targets=targets,
            token_scoring_gt_generator=token_scoring_gt_generator,
        )
        sgdt_output_list += sgdt_output_list_decoder
        return hs, references, sgdt_output_list


# ------------------
# TTI modification
# ------------------
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer_list, norm=None, d_model=256,
                 ):
        super().__init__()

        # self.layers = _get_clones(encoder_layer, num_layers)
        # self.num_layers = num_layers

        self.query_scale = MLP(d_model, d_model, d_model, 2)

        self.norm = norm

        # ------------------- TTI Modification
        self.layers = nn.ModuleList()
        self.num_layers = 0
        for l_conf in encoder_layer_list:
            encoder_layer, num_l = l_conf
            assert num_l > 0
            # nn.ModuleList
            self.layers.extend(_get_clones(encoder_layer, num_l))
            self.num_layers += num_l

        # self.with_sgdt_layer = False
        # if encoder_sgdt_layer is not None and num_encoder_sgdt_layers > 0:
        #     self.sgdt_layers = _get_clones(encoder_sgdt_layer, num_encoder_sgdt_layers)
        #     self.with_sgdt_layer = True
        # self.sgdt = SGDT_module(embed_dim=d_model)
        # -------------------

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt_targets=None,  # only for debugging
                feat_map_size=None,
                sigma=None,
                ):
        # mask is always None, all others are not None.
        # the src_mask is only used by src_key_padding_mask (not mask)
        output = src  # torch.Size([800, 2, 256])

        sgdt_output_list = []
        for layer_id, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoderLayer):  # Regular TransformerEncoderLayer

                # rescale the content and pos sim
                #  to obtain a scale vector conditional on the content information and use it perform
                #  element-wise multiplication with the positional embeddings (first introduced in the code of DAB-DETR,
                #  not in Conditional DETR, not int DETR)
                pos_scales = self.query_scale(output)

                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos * pos_scales)

            elif isinstance(layer, (TransformerEncoderSGDTLayerV1,
                                    TransformerEncoderSGDTLayer)):
                # SGDT layers (v0, v1) used for token adaption, input: a set of tokens, output: a set of tokens.

                # In sgdt layer (v0), the pos_scaling should be disabled, because the padded tokens
                # used for calculate scaling factor is random (not trained), thus the learned scale for
                # those tokens are not reliable. Thus the best way is to disable the scaling for all tokens
                # no matter it is valid tokens or invalid tokens.

                token_scoring_config_parser = layer.sgdt.token_scoring_config_parser
                if token_scoring_config_parser is not None and \
                        token_scoring_config_parser.reclaim_padded_region:  #
                    # do not modify original pos if reclaim_padded_region = True
                    pos_scales = torch.ones_like(output)  # TODO: maybe we do not need this.
                else:
                    pos_scales = self.query_scale(output)  # torch.Size([713, 2, 256])

                output, sgdt_output = layer(output, src_mask=mask,  # sgdt=self.sgdt,
                                            src_key_padding_mask=src_key_padding_mask,
                                            pos=pos * pos_scales,
                                            sgdt_targets=sgdt_targets,
                                            feat_map_size=feat_map_size,  # feat_map_size = (h, w)
                                            sigma=sigma,
                                            )

                # need to update pos, if sgdt v0 layer is applied.
                # and not self.encoder_without_pos:  # no need to check encoder_without_pos
                if 'adapted_pos' in sgdt_output and sgdt_output['adapted_pos']:  #
                    # adapted original pos_embed, here we should not use pos=pos * pos_scales
                    # as the original pos is expected for later layer and decoder, each pos_scales are applied
                    # to original pos.
                    pos = extract_adapted_token_pos_embed(sgdt_output, pos=pos)

                # need to update mask, if sgdt v0 layer is applied and reclaim_padded_region_ is True.
                # sgdt v1 layer will have  src_mask_reclaimed = False even reclaim_padded_region_ is True
                # because query are the original tokens, only k include reclaim_padded_region
                if 'src_mask_reclaimed' in sgdt_output and sgdt_output['src_mask_reclaimed'] is not None:
                    # some tokens are reclaimed so they are not invalid tokens any more, we need to
                    # adapt the invalid token mask as well.
                    # ** ATTENTION **, if this section is put in Transformer, we should update mask,
                    # but if it is inside TransformerEncoder, we should update the src_key_padding_mask.
                    # mask = sgdt_output['src_mask_reclaimed']  # When in Transformer
                    src_key_padding_mask = sgdt_output['src_mask_reclaimed']

                # save the feat_map_size for debugging purpose
                sgdt_output['feat_map_size'] = torch.tensor(feat_map_size).to(output.device)
                sgdt_output_list.append(sgdt_output)
                # -------------------

        if self.norm is not None:
            output = self.norm(output)

        return output, sgdt_output_list, pos, src_key_padding_mask  # Note: not pass mask


class TransformerSGDTEncoder(TransformerEncoder):

    def forward_encoder_layers(self, encoder_layers, src,
                               mask: Optional[Tensor] = None,
                               src_key_padding_mask: Optional[Tensor] = None,
                               pos: Optional[Tensor] = None,
                               sgdt_targets=None,  # only for debugging
                               feat_map_size=None,
                               sigma=None,
                               ):
        output = src  # torch.Size([800, 2, 256])

        sgdt_output_list = []
        for layer_id, layer in enumerate(encoder_layers):
            if isinstance(layer, TransformerEncoderLayer):  # Regular TransformerEncoderLayer

                # rescale the content and pos sim
                #  to obtain a scale vector conditional on the content information and use it perform
                #  element-wise multiplication with the positional embeddings (first introduced in the code of DAB-DETR,
                #  not in Conditional DETR, not int DETR)
                pos_scales = self.query_scale(output)

                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos * pos_scales)

            elif isinstance(layer, (TransformerEncoderSGDTLayerV1,
                                    TransformerEncoderSGDTLayer)):
                # SGDT layers (v0, v1) used for token adaption, input: a set of tokens, output: a set of tokens.

                # In sgdt layer (v0), the pos_scaling should be disabled, because the padded tokens
                # used for calculate scaling factor is random (not trained), thus the learned scale for
                # those tokens are not reliable. Thus the best way is to disable the scaling for all tokens
                # no matter it is valid tokens or invalid tokens.

                token_scoring_config_parser = layer.sgdt.token_scoring_config_parser
                if token_scoring_config_parser is not None and \
                        token_scoring_config_parser.reclaim_padded_region:  #
                    # do not modify original pos if reclaim_padded_region = True
                    pos_scales = torch.ones_like(output)  # TODO: maybe we do not need this.
                else:
                    pos_scales = self.query_scale(output)  # torch.Size([713, 2, 256])

                output, sgdt_output = layer(output, src_mask=mask,  # sgdt=self.sgdt,
                                            src_key_padding_mask=src_key_padding_mask,
                                            pos=pos * pos_scales,
                                            sgdt_targets=sgdt_targets,
                                            feat_map_size=feat_map_size,  # feat_map_size = (h, w)
                                            sigma=sigma,
                                            )

                # need to update pos, if sgdt v0 layer is applied.
                # and not self.encoder_without_pos:  # no need to check encoder_without_pos
                if 'adapted_pos' in sgdt_output and sgdt_output['adapted_pos']:  #
                    # adapted original pos_embed, here we should not use pos=pos * pos_scales
                    # as the original pos is expected for later layer and decoder, each pos_scales are applied
                    # to original pos.
                    pos = extract_adapted_token_pos_embed(sgdt_output, pos=pos)

                # need to update mask, if sgdt v0 layer is applied and reclaim_padded_region_ is True.
                # sgdt v1 layer will have  src_mask_reclaimed = False even reclaim_padded_region_ is True
                # because query are the original tokens, only k include reclaim_padded_region
                if 'src_mask_reclaimed' in sgdt_output and sgdt_output['src_mask_reclaimed'] is not None:
                    # some tokens are reclaimed so they are not invalid tokens any more, we need to
                    # adapt the invalid token mask as well.
                    # ** ATTENTION **, if this section is put in Transformer, we should update mask,
                    # but if it is inside TransformerEncoder, we should update the src_key_padding_mask.
                    # mask = sgdt_output['src_mask_reclaimed']  # When in Transformer
                    src_key_padding_mask = sgdt_output['src_mask_reclaimed']

                # save the feat_map_size for debugging purpose
                sgdt_output['feat_map_size'] = torch.tensor(feat_map_size).to(output.device)
                sgdt_output_list.append(sgdt_output)
                # -------------------

        assert self.norm is None, 'pre-norm is not support, as we the memory passed to the decoder will ' \
                                  'conduct norm again in the last layer.'

        return output, sgdt_output_list, pos, src_key_padding_mask  # Note: not pass mask

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt_targets=None,  # only for debugging
                feat_map_size=None,

                forward_layers='all',  # 'first_half',  'second_half'
                sigma=None,
                ):

        if forward_layers == 'first_half':
            encoder_layers = self.layers[:-1]
        elif forward_layers == 'second_half':  # currently only support the last layer
            encoder_layers = self.layers[-1:]
        elif forward_layers == 'all':
            encoder_layers = self.layers
        else:
            raise NotImplementedError

        output, sgdt_output_list, pos, src_key_padding_mask = self.forward_encoder_layers(
            encoder_layers=encoder_layers,
            src=src, mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            pos=pos,
            sgdt_targets=sgdt_targets, feat_map_size=feat_map_size,
            sigma=sigma,
        )

        # if self.norm is not None and forward_layers != 'first_half':
        #     output = self.norm(output)

        return output, sgdt_output_list, pos, src_key_padding_mask


class TransformerSGDTDecoder(TransformerDecoder):

    # def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=256, query_dim=2,
    #              keep_query_pos=False, query_scale_type='cond_elewise', modulate_hw_attn=False,
    #              bbox_embed_diff_each_layer=False,
    #
    #              encoder_layer_list=None,
    #              encoder_query_scale=None,
    #              ):
    # super().__init__(decoder_layer, num_layers, norm, return_intermediate, d_model, query_dim, keep_query_pos,
    #                  query_scale_type, modulate_hw_attn, bbox_embed_diff_each_layer)

    # # ------------------- TTI Modification for handling sgdt encoder layers inside the decoder.
    # self.num_encoder_layers = 0
    # self.encoder_layers = nn.ModuleList()
    # self.encoder_query_scale = encoder_query_scale
    # if encoder_layer_list is not None:
    #     for l_conf in encoder_layer_list:
    #         encoder_layer, num_l = l_conf
    #         assert num_l > 0
    #         self.encoder_layers.extend(_get_clones(encoder_layer, num_l))
    #         self.num_encoder_layers += num_l

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2

                sgdt_targets=None,  # only for debugging
                feat_map_size=None,
                encoder=None,
                encoder_without_pos=False,
                sigma=None,
                mask_dict=None,  # mask_dict=mask_dict,
                proposal_processor=None,
                class_embed=None,  # class_embed=class_embed
                input_img_sizes=None,  # input_img_sizes=input_img_sizes,
                targets=None,
                token_scoring_gt_generator=None,
                ):
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        # import ipdb; ipdb.set_trace()
        sgdt_output_list = []
        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]  # [num_queries, batch_size, 2]
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)

            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation
            query_sine_embed = query_sine_embed[..., :self.d_model] * pos_transformation

            # modulated HW attentions
            if self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)

            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))

            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

            # --------------------- update the encoder features here (just before the last decoder layer)
            #  memory, memory_key_padding_mask, pos might be updated in below lines (memory_mask is always None)
            # Below is how the decoder is called.
            # hs, references = self.decoder(tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
            #                               pos=pos_embed, refpoints_unsigmoid=refpoint_embed)
            if encoder is not None and layer_id == len(self.layers) - 2:
                if proposal_processor is not None:
                    # targets = get_targets_from_proposals(selected_proposals, targets)
                    selected_proposals = self._extract_proposals(intermediate, ref_points, reference_points,
                                                                 mask_dict, input_img_sizes, proposal_processor,
                                                                 class_embed=class_embed,
                                                                 )
                    #                 class_embed=None,   # class_embed=class_embed
                    targets = update_targets_with_proposals(selected_proposals, targets)
                    sgdt_target_raw = token_scoring_gt_generator.get_gt_raw(targets=targets)
                    sgdt_targets = token_scoring_gt_generator.resize_sig_value_gt(
                        sgdt_target_raw, feat_map_size=feat_map_size)

                memory, sgdt_output_list, pos, memory_key_padding_mask = encoder(
                    src=memory, src_key_padding_mask=memory_key_padding_mask,
                    pos=pos if not encoder_without_pos else torch.zeros_like(pos),
                    sgdt_targets=sgdt_targets,
                    feat_map_size=feat_map_size,
                    forward_layers='second_half',  # 'all', 'first_half',  'second_half'
                    sigma=sigma,
                )

            # --------------------------------------

        if self.norm is not None:
            output = self.norm(output)  # output torch.Size([300, 2, 256])
            if self.return_intermediate:
                intermediate.pop()  # the predictions of intermediate layers are here.
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                    sgdt_output_list
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2),
                    sgdt_output_list
                ]

        return output.unsqueeze(0), sgdt_output_list

    def _extract_proposals(self, intermediate, ref_points, reference_points,
                           mask_dict, input_img_sizes, proposal_processor, class_embed):
        assert self.return_intermediate

        if self.bbox_embed is not None:
            num_out = len(intermediate)
            hs, reference = torch.stack(intermediate).transpose(1, 2), \
                            torch.stack(ref_points[:num_out]).transpose(1, 2)  # ref_points is a list of 6 items
        else:
            hs, reference = torch.stack(intermediate).transpose(1, 2), \
                            reference_points.unsqueeze(0).transpose(1, 2)

        if not self.bbox_embed_diff_each_layer:
            reference_before_sigmoid = inverse_sigmoid(reference)  # torch.Size([5, 2, 405, 4])
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

        outputs_class = class_embed(hs)
        # dn post process
        outputs_class, outputs_coord = dn_post_process(outputs_class, outputs_coord, mask_dict)
        proposals = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        # selected_proposals = proposal_processor.top_proposals(proposals, target_sizes=orig_target_sizes)
        selected_proposals = proposal_processor(proposals, target_sizes=input_img_sizes)
        return selected_proposals


class TransformerEmptyEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        src = self.with_pos_embed(src, pos)
        sgdt_output_list = []
        return src, sgdt_output_list


class TransformerEncoderSGDTLayer(nn.Module):
    # replace the self attention with dynamic attention from TCFormer, and FFN.

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 # reclaim_padded_region=False,
                 token_scoring_discard_split_criterion=None,
                 ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # --------------------------------
        self.sgdt = SGDT_module(embed_dim=d_model,
                                token_scoring_discard_split_criterion=token_scoring_discard_split_criterion,
                                )
        # Token adaption module, shared across all layers

        # self.reclaim_padded_region = reclaim_padded_region
        self.nhead = nhead
        # --------------------------------

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt_targets=None,
                feat_map_size=None,  # feat_map_size = (h, w)
                sigma=None,
                ):
        """

        Args:
            src:
            src_mask:  attention mask.
            src_key_padding_mask:
            pos:
            sgdt_targets:

        Returns:

        """
        # here q, k from different source. # -----
        # q = k = self.with_pos_embed(src, pos)

        k = self.with_pos_embed(src, pos)  # k, v are from the original tokens.

        # ------------------------
        sgdt_output = \
            self.sgdt(src, mask=src_key_padding_mask,
                      sgdt_targets=sgdt_targets,
                      feat_map_size=feat_map_size,  # feat_map_size = (h, w)
                      sigma=sigma,
                      # reclaim_padded_region=self.reclaim_padded_region
                      )  # torch.Size([630, 2, 256]),torch.Size([2, 630])

        # # ################################### wrong version
        # # important modification here, we use adapted_token_dict as q_dict, so
        # # short path should be adapted_token_dict instead of the original token_dict (kv_dict)
        # # value should not be orignal src, but the adapted tokens.
        # src = sgdt_output['x']
        #
        # q_adapted_pos = extract_adapted_token_pos_embed(adapted_token_dict=sgdt_output, pos=pos)
        # q = self.with_pos_embed(src, pos=q_adapted_pos)  # using adapted tokens as queries
        #
        # # tcformer_layers.py forward(self, q_dict, kv_dict): so k v should from the same src.
        # #  but here we q, v are from the same src.
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        # src = src + self.dropout1(src2)
        #
        # src = self.norm1(src)
        # # ###################################

        q_adapted_pos = extract_adapted_token_pos_embed(adapted_token_dict=sgdt_output, pos=pos)
        q = self.with_pos_embed(sgdt_output['x'], pos=q_adapted_pos)  # using adapted tokens as queries

        # tcformer_layers.py forward(self, q_dict, kv_dict): so k v should come from the same src.
        #  but here we q, v are from the same src, because the attention means the attention to the key
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        # TODO: check if I need to use src or sgdt_output['x'] (maybe TCformer used sgdt_output['x']?)
        src = sgdt_output['x'] + self.dropout1(src2)

        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # TODO: replace with dwconv as in TCFormer
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # pass the related infor of sgdt module for loss calculation and pos adaption.

        # The input pos is the scaled pos, not the original pos, so we do not pass q_adapted_pos
        # out, but just return a flag saying that the original pos should be adapted outside this
        # function with the original pos..
        sgdt_output.update(dict(adapted_pos=True))

        return src, sgdt_output


class TransformerEncoderSGDTLayerV1(TransformerEncoderSGDTLayer):
    # Q from original token, k, v from adapted tokens.

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt_targets=None,
                feat_map_size=None,  # feat_map_size = (h, w) =
                sigma=None,
                ):
        """
        2022-10-3 fixed two bugs, 1) v to be same with updated k, 2) src_mask_reclaimed set to None (mask should not
        be updated in decoder).
        Args:
            src:
            src_mask:  attention mask.
            src_key_padding_mask:
            pos:
            sgdt_targets:

        Returns:
        # generate attention mask to disable the split two tokens to see each other.
        # Suppose Token k is split to k1, k2; J is split to J1 and J2, then k1 should not see k2, J1 should
        # not see J2, but k1 can see J1, J2, as they represent different spatial locations.

        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        """
        # here q, k from different source. # -----
        # q = k = self.with_pos_embed(src, pos
        q = self.with_pos_embed(src, pos)  # k, v are from the original tokens.
        sgdt_output = self.sgdt(src, mask=src_key_padding_mask,
                                sgdt_targets=sgdt_targets,
                                feat_map_size=feat_map_size,  # (h, w)
                                sigma=sigma,
                                )

        k_adapted_pos = extract_adapted_token_pos_embed(
            adapted_token_dict=sgdt_output, pos=pos)
        k = self.with_pos_embed(sgdt_output['x'], pos=k_adapted_pos)  # using adapted tokens as queries

        # I should update the key padding mask here, but delete the mask to avoid it to be
        # used outside the sgdt layer.')
        if 'src_mask_reclaimed' in sgdt_output and sgdt_output['src_mask_reclaimed'] is not None:
            # some tokens are reclaimed so they are not invalid tokens any more, we need to
            # adapt the invalid token mask as well.
            # ** ATTENTION **, if this section is put in Transformer, we should update mask,
            # but if it is inside TransformerEncoder, we should update the src_key_padding_mask.
            # mask = sgdt_output['src_mask_reclaimed']  # When in Transformer
            src_key_padding_mask = sgdt_output['src_mask_reclaimed']

        # value is not src anymore, as k has been updated.
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(q, k, value=sgdt_output['x'], attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        src = self.norm1(src)
        # TODO: replace with dwconv as in TCFormer
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # the src_mask_reclaimed should not be used outside this encoder layer to update src_key_padding_mask,
        # as query is not updated (the update of k, v will not change the src_key_padding_mask).
        sgdt_output.update(dict(adapted_pos=False, src_mask_reclaimed=None))

        return src, sgdt_output


def build_transformer(args):
    if args.sgdt_transformer:
        transformer = TransformerSGDT
    else:
        transformer = Transformer

    return transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        # num_encoder_layers=args.enc_layers,

        # --------------------
        # num_encoder_sgdt_layers=args.num_encoder_sgdt_layers,
        encoder_without_pos=args.encoder_without_pos,
        # encoder_sgdt_layer_version=args.encoder_sgdt_layer_version,
        # reclaim_padded_region=args.reclaim_padded_region,
        token_scoring_discard_split_criterion=args.token_scoring_discard_split_criterion,
        encoder_layer_config=args.encoder_layer_config,
        # --------------------

        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=4,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        # train_token_scoring_only=args.train_token_scoring_only
    )

