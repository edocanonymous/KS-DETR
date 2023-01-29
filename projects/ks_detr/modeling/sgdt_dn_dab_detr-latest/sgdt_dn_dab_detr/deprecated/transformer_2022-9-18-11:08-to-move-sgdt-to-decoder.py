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
from .attention import MultiheadAttention

# ------------------- TTI Modification
from models.sgdt.sgdt_module import SGDT_module, get_valid_token_mask, TokenScoringConfigParser
from models.sgdt.scoring_gt import resize_sgdt_target


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class Transformer(nn.Module):

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
                 encoder_layer_config='regular_6'
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
            self.encoder = TransformerEncoder(encoder_layer_list, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos,
                                          query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

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

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, refpoint_embed, pos_embed, tgt, attn_mask=None,
                sgdt_target_raw=None  # for debugging only
                ):
        # flatten NxCxHxW to HWxNxC # h= 25, w = 32
        bs, c, h, w = src.shape  # torch.Size([2, 256, 25, 32])
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # torch.Size([2, 256, 25, 32]) -> torch.Size([800, 2, 256])
        # refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)
        # -------------------------
        sgdt_targets = resize_sgdt_target(
            sgdt_target_raw, feat_map_size=(h, w), feat_map_mask=mask)
        # mask, (B, H, W)  -> (B, N), e.g., torch.Size([2, 30, 23]) -> torch.Size([2, 690])
        mask = mask.flatten(1)  # torch.Size([2, 21, 30]) -> torch.Size([2, 630]), src = torch.Size([630, 2, 256])

        # do not accept the adapted pos_embed, mask from encoder as zeros pos_embed might be fed
        memory, sgdt_output_list, pos, mask = self.encoder(
            src, src_key_padding_mask=mask,
            pos=pos_embed if not self.encoder_without_pos else torch.zeros_like(pos_embed),
            sgdt_targets=sgdt_targets,
            feat_map_size=(h, w)
        )

        if self.num_patterns > 0:
            l = tgt.shape[0]
            tgt[l - self.num_queries * self.num_patterns:] += \
                self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1)

        hs, references = self.decoder(tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
                                      pos=pos_embed, refpoints_unsigmoid=refpoint_embed)
        return hs, references, sgdt_output_list


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
                 encoder_layer_config='regular_6'
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

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, refpoint_embed, pos_embed, tgt, attn_mask=None,
                sgdt_target_raw=None  # for debugging only
                ):
        # flatten NxCxHxW to HWxNxC # h= 25, w = 32
        bs, c, h, w = src.shape  # torch.Size([2, 256, 25, 32])
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # torch.Size([2, 256, 25, 32]) -> torch.Size([800, 2, 256])
        # refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)
        # -------------------------
        sgdt_targets = resize_sgdt_target(
            sgdt_target_raw, feat_map_size=(h, w), feat_map_mask=mask)
        # mask, (B, H, W)  -> (B, N), e.g., torch.Size([2, 30, 23]) -> torch.Size([2, 690])
        mask = mask.flatten(1)  # torch.Size([2, 21, 30]) -> torch.Size([2, 630]), src = torch.Size([630, 2, 256])

        # do not accept the adapted pos_embed, mask from encoder as zeros pos_embed might be fed
        # memory, sgdt_output_list = self.encoder(
        #     src, src_key_padding_mask=mask,
        #     pos=pos_embed if not self.encoder_without_pos else torch.zeros_like(pos_embed),
        #     sgdt_targets=sgdt_targets,
        #     feat_map_size=(h, w)
        # )

        memory, sgdt_output_list, pos, mask = self.encoder(
            src, src_key_padding_mask=mask,
            pos=pos_embed if not self.encoder_without_pos else torch.zeros_like(pos_embed),
            sgdt_targets=sgdt_targets,
            feat_map_size=(h, w),
            forward_layers='first_half',  # 'all', 'first_half',  'second_half'
        )

        if self.num_patterns > 0:
            l = tgt.shape[0]
            tgt[l - self.num_queries * self.num_patterns:] += \
                self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1)

        # hs, references = self.decoder(tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
        #                               pos=pos_embed, refpoints_unsigmoid=refpoint_embed)

        hs, references, sgdt_output_list = self.decoder(tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
                                                        pos=pos_embed, refpoints_unsigmoid=refpoint_embed,

                                                        encoder=self.encoder,
                                                        )

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
                ):
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
                    pos_scales = torch.ones_like(output)
                else:
                    pos_scales = self.query_scale(output)  # torch.Size([713, 2, 256])

                output, sgdt_output = layer(output, src_mask=mask,  # sgdt=self.sgdt,
                                            src_key_padding_mask=src_key_padding_mask,
                                            pos=pos * pos_scales,
                                            sgdt_targets=sgdt_targets,
                                            feat_map_size=feat_map_size,  # feat_map_size = (h, w)
                                            )

                # need to update pos, if sgdt v0 layer is applied.
                # and not self.encoder_without_pos:  # no need to check encoder_without_pos
                if 'adapted_pos' in sgdt_output and sgdt_output['adapted_pos']:  #
                    # adapted original pos_embed
                    pos = extract_adapted_token_pos_embed(sgdt_output, pos=pos)

                # need to update mask, if sgdt v0 layer is applied and reclaim_padded_region_ is True.
                # sgdt v1 layer will have  src_mask_reclaimed = False even reclaim_padded_region_ is True
                # because query are the original tokens, only k include reclaim_padded_region
                if 'src_mask_reclaimed' in sgdt_output and sgdt_output['src_mask_reclaimed'] is not None:
                    # some tokens are reclaimed so they are not invalid tokens any more, we need to
                    # adapt the invalid token mask as well.
                    mask = sgdt_output['src_mask_reclaimed']

                # save the feat_map_size for debugging purpose
                sgdt_output['feat_map_size'] = torch.tensor(feat_map_size).to(output.device)
                sgdt_output_list.append(sgdt_output)
                # -------------------

        if self.norm is not None:
            output = self.norm(output)

        return output, sgdt_output_list, pos, mask


class TransformerSGDTEncoder(TransformerEncoder):

    def forward_encoder_layers(self, encoder_layers, src,
                               mask: Optional[Tensor] = None,
                               src_key_padding_mask: Optional[Tensor] = None,
                               pos: Optional[Tensor] = None,
                               sgdt_targets=None,  # only for debugging
                               feat_map_size=None,

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
                    pos_scales = torch.ones_like(output)
                else:
                    pos_scales = self.query_scale(output)  # torch.Size([713, 2, 256])

                output, sgdt_output = layer(output, src_mask=mask,  # sgdt=self.sgdt,
                                            src_key_padding_mask=src_key_padding_mask,
                                            pos=pos * pos_scales,
                                            sgdt_targets=sgdt_targets,
                                            feat_map_size=feat_map_size,  # feat_map_size = (h, w)
                                            )

                # need to update pos, if sgdt v0 layer is applied.
                # and not self.encoder_without_pos:  # no need to check encoder_without_pos
                if 'adapted_pos' in sgdt_output and sgdt_output['adapted_pos']:  #
                    # adapted original pos_embed
                    pos = extract_adapted_token_pos_embed(sgdt_output, pos=pos)

                # need to update mask, if sgdt v0 layer is applied and reclaim_padded_region_ is True.
                # sgdt v1 layer will have  src_mask_reclaimed = False even reclaim_padded_region_ is True
                # because query are the original tokens, only k include reclaim_padded_region
                if 'src_mask_reclaimed' in sgdt_output and sgdt_output['src_mask_reclaimed'] is not None:
                    # some tokens are reclaimed so they are not invalid tokens any more, we need to
                    # adapt the invalid token mask as well.
                    mask = sgdt_output['src_mask_reclaimed']

                # save the feat_map_size for debugging purpose
                sgdt_output['feat_map_size'] = torch.tensor(feat_map_size).to(output.device)
                sgdt_output_list.append(sgdt_output)
                # -------------------

        return output, sgdt_output_list, pos, mask

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt_targets=None,  # only for debugging
                feat_map_size=None,

                forward_layers='all',  # 'first_half',  'second_half'
                ):

        if forward_layers == 'first_half':
            encoder_layers = self.layers[:-1]
        elif forward_layers == 'second_half':  # currently only support the last layer
            encoder_layers = self.layers[-1:]
        elif forward_layers == 'all':
            encoder_layers = self.layers
        else:
            raise NotImplementedError

        output, sgdt_output_list, pos, mask = self.forward_encoder_layers(
            encoder_layers=encoder_layers,
            src=src, mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            pos=pos,
            sgdt_targets=sgdt_targets,  feat_map_size=feat_map_size)

        if self.norm is not None and forward_layers != 'first_half':
            output = self.norm(output)

        return output, sgdt_output_list, pos, mask


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                 modulate_hw_attn=False,
                 bbox_embed_diff_each_layer=False,
                 ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)

        self.bbox_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                ):
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        # import ipdb; ipdb.set_trace()        

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

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        return output.unsqueeze(0)


class TransformerSGDTDecoder(TransformerDecoder):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=256, query_dim=2,
                 keep_query_pos=False, query_scale_type='cond_elewise', modulate_hw_attn=False,
                 bbox_embed_diff_each_layer=False,

                 encoder_layer_list=None,
                 encoder_query_scale=None,
                 ):
        super().__init__(decoder_layer, num_layers, norm, return_intermediate, d_model, query_dim, keep_query_pos,
                         query_scale_type, modulate_hw_attn, bbox_embed_diff_each_layer)

        # ------------------- TTI Modification for handling sgdt encoder layers inside the decoder.
        self.num_encoder_layers = 0
        self.encoder_layers = nn.ModuleList()
        self.encoder_query_scale = encoder_query_scale
        if encoder_layer_list is not None:
            for l_conf in encoder_layer_list:
                encoder_layer, num_l = l_conf
                assert num_l > 0
                self.encoder_layers.extend(_get_clones(encoder_layer, num_l))
                self.num_encoder_layers += num_l

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2

                sgdt_targets=None,  # only for debugging
                feat_map_size=None,
                ):
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        # import ipdb; ipdb.set_trace()

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
            if self.num_encoder_layers > 0 and layer_id == len(self.layers) - 2:
                #  memory, memory_key_padding_mask, pos might be updated in below lines (memory_mask is always None)
                # Below is how the decoder is called.
                # hs, references = self.decoder(tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
                #                               pos=pos_embed, refpoints_unsigmoid=refpoint_embed)

                # from encoder
                sgdt_output_list = []
                for encoder_layer_id, encoder_layer in enumerate(self.encoder_layers):
                    if isinstance(encoder_layer, TransformerEncoderLayer):  # Regular TransformerEncoderLayer

                        pos_scales = self.encoder_query_scale(memory)  # the query_scale layer in encoder.

                        memory = encoder_layer(memory, src_mask=memory_mask,
                                               src_key_padding_mask=memory_key_padding_mask,
                                               pos=pos * pos_scales)

                    elif isinstance(encoder_layer, (TransformerEncoderSGDTLayerV1,
                                                    TransformerEncoderSGDTLayer)):
                        # SGDT layers (v0, v1) used for token adaption, input: a set of tokens, output: a set of tokens.

                        # In sgdt layer (v0), the pos_scaling should be disabled, because the padded tokens
                        # used for calculate scaling factor is random (not trained), thus the learned scale for
                        # those tokens are not reliable. Thus the best way is to disable the scaling for all tokens
                        # no matter it is valid tokens or invalid tokens.

                        token_scoring_config_parser = encoder_layer.sgdt.token_scoring_config_parser
                        if token_scoring_config_parser is not None and \
                                token_scoring_config_parser.reclaim_padded_region:  #
                            # do not modify original pos if reclaim_padded_region = True
                            pos_scales = torch.ones_like(memory)
                        else:
                            pos_scales = self.encoder_query_scale(memory)  # torch.Size([713, 2, 256])

                        # TODO: update the sgdt_targets from the output
                        # proposals = output
                        memory, sgdt_output = encoder_layer(memory, src_mask=memory_mask,  # sgdt=self.sgdt,
                                                            src_key_padding_mask=memory_key_padding_mask,
                                                            pos=pos * pos_scales,
                                                            sgdt_targets=sgdt_targets,
                                                            feat_map_size=feat_map_size,  # feat_map_size = (h, w)
                                                            )

                        # Do not adapt the pos and mask for the last layer as they will not be further used.
                        # if encoder_layer_id != len(self.encoder_layers) - 1:
                        # the modified pos and memory_key_padding_mask will be used in the later decoder layers.

                        # need to update pos, if sgdt v0 layer is applied.
                        # and not self.encoder_without_pos:  # no need to check encoder_without_pos
                        if 'adapted_pos' in sgdt_output and sgdt_output['adapted_pos']:  #
                            # adapted original pos_embed
                            pos = extract_adapted_token_pos_embed(sgdt_output, pos=pos)

                        # need to update mask, if sgdt v0 layer is applied and reclaim_padded_region_ is True.
                        # sgdt v1 layer will have  src_mask_reclaimed = False even reclaim_padded_region_ is True
                        # because query are the original tokens, only k include reclaim_padded_region
                        if 'src_mask_reclaimed' in sgdt_output and sgdt_output['src_mask_reclaimed'] is not None:
                            # some tokens are reclaimed so they are not invalid tokens any more, we need to
                            # adapt the invalid token mask as well.
                            memory_key_padding_mask = sgdt_output['src_mask_reclaimed']

                        sgdt_output_list.append(sgdt_output)
                        # -------------------

            # --------------------------------------

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        return output.unsqueeze(0), sgdt_output_list


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


def extract_adapted_token_pos_embed(adapted_token_dict, pos: Optional[Tensor]):
    """
    return extracted pos based on tokens_small_obj, tokens_to_discard in adapted_token_dict
    Args:
        adapted_token_dict: a dict included tokens_small_obj, tokens_to_discard
        pos: learnable position_embedding, (N, B, C), e.g., torch.Size([800, 2, 256])
    Returns:
    """
    if pos is None:
        return pos

    else:
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
        # number to discard = number of small (for each image)
        # N, B, bool(), 1, small ->
        tokens_small_obj = adapted_token_dict['tokens_small_obj'].unsqueeze(-1).repeat(1, 1, C)
        # N, B; bool(), 1, to discard
        tokens_to_discard = adapted_token_dict['tokens_to_discard'].unsqueeze(-1).repeat(1, 1, C)
        # TODO: do we need to detach? maybe not. .detach()
        adapted_pos = pos.clone()  # (N, B, C), e.g., torch.Size([800, 2, 256])  # pos: requires_grad = True grad
        # TODO: directly modify the value, differentiable?
        adapted_pos[tokens_to_discard] = pos[tokens_small_obj]

        return adapted_pos


class TransformerEncoderSGDTLayer(nn.Module):
    # replace the self attentionwith dynamic attention from TCFormer, and FFN.

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
                                token_scoring_discard_split_criterion=token_scoring_discard_split_criterion)
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

        # ------------------------

        q = self.with_pos_embed(src, pos)  # k, v are from the original tokens.
        # ------------------------

        """
        token_dict = {'x': x,
              'token_num': N,
              'map_size': [H, W],
              'init_grid_size': [H, W],
              'idx_token': idx_token,
              'agg_weight': agg_weight}
        """
        # adapted_token_dict, fg_score_logit, small_scale_score_logit, valid_tokens = \
        #     self.sgdt(src, mask=src_key_padding_mask,
        #               sgdt_targets=sgdt_targets,
        #               feat_map_size=feat_map_size,  # feat_map_size = (h, w)
        #               )  # torch.Size([630, 2, 256]),torch.Size([2, 630])
        sgdt_output = self.sgdt(src, mask=src_key_padding_mask,
                                sgdt_targets=sgdt_targets,
                                feat_map_size=feat_map_size,  # (h, w)
                                )

        k_adapted_pos = extract_adapted_token_pos_embed(
            adapted_token_dict=sgdt_output, pos=pos)

        # ---------------------
        k = self.with_pos_embed(sgdt_output['x'], pos=k_adapted_pos)  # using adapted tokens as queries
        # ---------------------

        # generate attention mask to disable the split two tokens to see each other.
        # Suppose Token k is split to k1, k2; J is split to J1 and J2, then k1 should not see k2, J1 should
        # not see J2, but k1 can see J1, J2, as they represent different spatial locations.

        # ------------------------------- no need this, as query are from adapted tokens, k, v are
        # the original tokens, the Q will attention all locations.
        """
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        """
        # -------------------
        # # src_mask
        # if src_mask is None:
        #     src_mask = adapted_token_dict['attn_mask']  # bool
        # else:
        #     raise NotImplementedError
        #     # logical and or or?

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # ---------------------
        # important modification here, we use adapted_token_dict as q_dict, so
        # short path should be adapted_token_dict instead of the original token_dict (kv_dict)
        src = src + self.dropout1(src2)
        # src = adapted_token_dict['x'] + self.dropout1(src2)
        # ---------------------

        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # TODO: replace with dwconv as in TCFormer
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        sgdt_output.update(dict(adapted_pos=False))

        return src, sgdt_output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False
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

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False):

        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)  # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                               key=k,
                               value=v, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
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
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
