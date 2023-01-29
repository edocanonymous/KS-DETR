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
    TransformerDecoderLayer, _get_activation_fn, _get_clones, gen_sineembed_for_position

from models.sgdt.sgdt_module import SGDT_module, get_valid_token_mask, TokenScoringConfigParser
# from models.sgdt.scoring_gt import resize_sgdt_target
from models.DN_DAB_DETR.dn_components import dn_post_process
from models.sgdt.scoring_gt import update_targets_with_proposals
from models.sgdt.sgdt_components import parser_encoder_layers
from models.sgdt.sgdt_ import SGDT


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
        # because pos_scale is differentiable, so pos.requires_grad could be True.
        # assert not pos.requires_grad, 'If use learnable position_embedding,
        # the code in extract_adapted_token_pos_embed' \
        #                               'should be adapted to be differentiable.'

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

        # else:
        #     # number to discard = number of small (for each image)
        #     # N, B, bool(), 1, small -> N, B, C
        #     tokens_small_obj = adapted_token_dict['tokens_small_obj'].unsqueeze(-1).repeat(1, 1, C)
        #     # N, B; bool(), 1, to discard -> N, B, C
        #     tokens_to_discard = adapted_token_dict['tokens_to_discard'].unsqueeze(-1).repeat(1, 1, C)
        #     # directly modify the value by index differentiable? Yes
        #     adapted_pos[tokens_to_discard] = pos[tokens_small_obj]  # adapted_pos.requires_grad = True

        return adapted_pos


class TransformerEmptyEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        src = self.with_pos_embed(src, pos)
        sgdt_output_list = []
        return src, sgdt_output_list, pos, src_key_padding_mask  # Note: not pass mask


class TransformerEncoderSGDTLayer(TransformerEncoderLayer):
    # Q from adapted tokens, k, v from original token

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,

                sgdt=None,
                # sgdt_targets=None,
                # feat_map_size=None,  # feat_map_size = (h, w)
                # sigma=None,
                ):
        """

        Args:
            sgdt:
            src:
            src_mask:  attention mask.
            src_key_padding_mask:
            pos:

        Returns:

        """
        # here q, k from different source. # -----
        # q = k = self.with_pos_embed(src, pos)
        k = self.with_pos_embed(src, pos)  # k, v are from the original tokens.
        # ------------------------
        sgdt_output = sgdt(x=src, mask=src_key_padding_mask,
                           # sgdt_targets=sgdt_targets,
                           # feat_map_size=feat_map_size,  # feat_map_size = (h, w)
                           # sigma=sigma,
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
        # Do not update src_key_padding_mask here, because k, v tokens are not adapted, adaption of q tokens will
        # will not affect the key_padding_mask.
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        # check if I need to use 'src' or sgdt_output['x'] (maybe TCformer used sgdt_output['x']?),
        # confirmed: TCformer used sgdt_output['x'] as the short path, i.e., use down_dict as src from
        # (down_dict, token_dict)
        # If use 'src' same as in SGDTV1, then the pos of the the tokens in 'src' and 'self.dropout1(src2)
        # are not consistent due the token adaption, so I should use sgdt_output['x']
        src = sgdt_output['x'] + self.dropout1(src2)

        src = self.norm1(src)
        # TODO: replace with dwconv as in TCFormer
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # Pass the related infor of sgdt module for loss calculation and pos adaption.
        # The input pos is the scaled pos, not the original pos, so we do not pass q_adapted_pos
        # out, but just return a flag saying that the original pos should be adapted outside this
        # function with the original pos..
        sgdt_output.update(dict(adapted_pos=True))

        # src_key_padding_mask can be safely updated here.
        # ** ATTENTION **, if this section is put in Transformer, we should update 'mask',
        # i.e., mask = sgdt_output['src_mask_reclaimed']
        # but if it is inside TransformerEncoder, we should update the 'src_key_padding_mask'.
        assert 'src_mask_reclaimed' in sgdt_output and sgdt_output['src_mask_reclaimed'] is not None
        src_key_padding_mask = sgdt_output['src_mask_reclaimed']

        return src, sgdt_output, src_key_padding_mask


class TransformerEncoderSGDTLayerV1(TransformerEncoderSGDTLayer):
    # Q from original token, k, v from adapted tokens.

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt=None,
                ):
        """
        2022-10-3 fixed two bugs, 1) v to be same with updated k, 2) src_mask_reclaimed set to None (mask should not
        be updated in decoder).
        Args:
            sgdt:
            src:
            src_mask:  attention mask.
            src_key_padding_mask:
            pos:

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
        sgdt_output = sgdt(x=src, mask=src_key_padding_mask,
                           # sgdt_targets=sgdt_targets,
                           # feat_map_size=feat_map_size,  # (h, w)
                           # sigma=sigma,
                           )
        # TODO: update this to make it more compilable.
        k_adapted_pos = extract_adapted_token_pos_embed(
            adapted_token_dict=sgdt_output, pos=pos)
        k = self.with_pos_embed(sgdt_output['x'], pos=k_adapted_pos)

        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]

        # 1. value is not src anymore, as k has been updated.
        # 2. some tokens are reclaimed so they are not invalid tokens any more, we need to
        # adapt the invalid token mask as well.
        assert 'src_mask_reclaimed' in sgdt_output and sgdt_output['src_mask_reclaimed'] is not None
        src2 = self.self_attn(q, k, value=sgdt_output['x'], attn_mask=src_mask,
                              key_padding_mask=sgdt_output['src_mask_reclaimed'])[0]
        src = src + self.dropout1(src2)

        src = self.norm1(src)
        # TODO: replace with dwconv as in TCFormer
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # k_adapted_pos is not used for self-attention inside the encoder layer,
        # should not be used outside Encoder layer
        sgdt_output.update(dict(adapted_pos=False))

        # the src_mask_reclaimed should not be used outside this encoder layer to update src_key_padding_mask,
        # as query is not updated (the update of k, v will not change the src_key_padding_mask).
        # I should not update the key padding mask here by src_key_padding_mask = sgdt_output['src_mask_reclaimed']
        # because it will not be used outside the sgdt layer

        return src, sgdt_output, src_key_padding_mask


SGDTEncoderLayerType = {
    'regular': TransformerEncoderLayer,
    'sgdt': TransformerEncoderSGDTLayer,
    'sgdtv1': TransformerEncoderSGDTLayerV1,
}


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

    def forward_subset_encoder_layers(self, src,
                                      mask: Optional[Tensor] = None,
                                      src_key_padding_mask: Optional[Tensor] = None,
                                      pos: Optional[Tensor] = None,

                                      encoder_layer_ids=None,
                                      sgdt=None,
                                      ):
        # mask is always None, all others are not None.
        # the src_mask is only used by src_key_padding_mask (not mask)
        output = src  # torch.Size([800, 2, 256])

        if encoder_layer_ids is None:
            encoder_layers = self.layers
        else:
            assert isinstance(encoder_layer_ids, list) and len(encoder_layer_ids) <= len(self.layers)
            encoder_layers = [self.layers[k] for k in encoder_layer_ids]

        sgdt_output_list = []
        encoder_output_list = []  # the output of each encoder layer
        for layer_id, layer in enumerate(encoder_layers):
            # rescale the content and pos sim
            #  to obtain a scale vector conditional on the content information and use it perform
            #  element-wise multiplication with the positional embeddings (first introduced in the code of DAB-DETR,
            #  not in Conditional DETR, not int DETR)

            # # the pos_encoding of the padded tokens will never be used, so we do not need to worry about
            # # if we will reclaim some of the padded tokens or not
            # # (token_scoring_config_parser.reclaim_padded_region)
            pos_scales = self.query_scale(output)  # torch.Size([713, 2, 256])

            # Do not use isinstance(layer, TransformerEncoderLayer), because TransformerEncoderSGDTLayer,
            # TransformerEncoderSGDTLayerV1 are also instance of TransformerEncoderLayer
            if type(layer).__name__ == 'TransformerEncoderLayer':  # Regular TransformerEncoderLayer
                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos * pos_scales)

            elif type(layer).__name__ in ['TransformerEncoderSGDTLayer', 'TransformerEncoderSGDTLayerV1']:
                # SGDT layers (v0, v1) used for token adaption
                output, sgdt_output, src_key_padding_mask = layer(
                    output, src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos=pos * pos_scales,  # scaled pos is passed in
                    sgdt=sgdt,
                )

                # We must update pos here instead of inside the layer forward() if sgdt v0 layer is applied.
                # Otherwise pos * pos_scales will be adapted not original pos.
                # It is the original pos is expected for later layer and decoder, each pos_scales are applied
                #  to original pos.
                if 'adapted_pos' in sgdt_output and sgdt_output['adapted_pos']:
                    pos = extract_adapted_token_pos_embed(sgdt_output, pos=pos)

                sgdt_output_list.append(sgdt_output)
                # -------------------

            encoder_output_list.append(output)
        # norm operations should be conducted in decoder not in encoder if each time we only conduct
        # forward operation on a subset of layers.

        # if self.norm is not None:
        #     output = self.norm(output)

        return output, sgdt_output_list, pos, src_key_padding_mask, encoder_output_list  # Note: not pass mask

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,

                sgdt=None,
                ):

        output, sgdt_output_list, pos, src_key_padding_mask, encoder_output_list = self.forward_subset_encoder_layers(
            src=src, mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos, sgdt=sgdt
        )

        if self.norm is not None:
            output = self.norm(output)

        return output, sgdt_output_list, pos, src_key_padding_mask, encoder_output_list  # Note: not pass mask

    # def forward(self, src,
    #             mask: Optional[Tensor] = None,
    #             src_key_padding_mask: Optional[Tensor] = None,
    #             pos: Optional[Tensor] = None,
    #
    #             sgdt=None,
    #             # sgdt_targets=None,  # only for debugging
    #             # feat_map_size=None,
    #             # sigma=None,
    #             ):
    #     # mask is always None, all others are not None.
    #     # the src_mask is only used by src_key_padding_mask (not mask)
    #     output = src  # torch.Size([800, 2, 256])
    #
    #     sgdt_output_list = []
    #     for layer_id, layer in enumerate(self.layers):
    #         # the pos_encoding of the padded tokens will never be used, so we do not need to worry about
    #         # if we will reclaim some of the padded tokens or not (token_scoring_config_parser.reclaim_padded_region)
    #
    #         # rescale the content and pos sim
    #         #  to obtain a scale vector conditional on the content information and use it perform
    #         #  element-wise multiplication with the positional embeddings (first introduced in the code of DAB-DETR,
    #         #  not in Conditional DETR, not int DETR)
    #         pos_scales = self.query_scale(output)  # torch.Size([713, 2, 256])
    #
    #         if isinstance(layer, TransformerEncoderLayer):  # Regular TransformerEncoderLayer
    #             output = layer(output, src_mask=mask,
    #                            src_key_padding_mask=src_key_padding_mask, pos=pos * pos_scales)
    #
    #         elif isinstance(layer, (TransformerEncoderSGDTLayer, TransformerEncoderSGDTLayerV1)):
    #             # SGDT layers (v0, v1) used for token adaption, input: a set of tokens, output: a set of tokens.
    #             output, sgdt_output, pos, src_key_padding_mask = layer(
    #                 output, src_mask=mask,
    #                 src_key_padding_mask=src_key_padding_mask,
    #                 pos=pos * pos_scales,
    #                 sgdt=sgdt,
    #             )
    #             sgdt_output_list.append(sgdt_output)
    #             # -------------------
    #
    #     if self.norm is not None:
    #         output = self.norm(output)
    #
    #     return output, sgdt_output_list, pos, src_key_padding_mask  # Note: not pass mask


class TransformerSGDTEncoder(TransformerEncoder):

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,

                encoder_layer_ids=None,
                sgdt=None,
                ):
        return self.forward_subset_encoder_layers(
            src=src, mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos, sgdt=sgdt,
            encoder_layer_ids=encoder_layer_ids
        )


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
                 encoder_layer_config='regular_6',
                 encoder_decoder_config=None,
                 align_encoder_decoder_layers_num=None,
                 ):

        super().__init__()

        # self.encoder_without_pos = encoder_without_pos

        if encoder_decoder_config is None:
            encoder_type, decoder_type = TransformerEncoder, TransformerDecoder
        elif encoder_decoder_config == 'sgdt':
            encoder_type, decoder_type = TransformerSGDTEncoder, TransformerSGDTDecoder
        elif encoder_decoder_config == 'self_distillation':
            encoder_type, decoder_type = TransformerSGDTEncoder, FeatureDistillationTransformerDecoder
        else:
            raise NotImplementedError

        # # self.reclaim_padded_region = reclaim_padded_region
        assert encoder_layer_config is not None and isinstance(encoder_layer_config, str)
        if len(encoder_layer_config) == 0:
            self.encoder = TransformerEmptyEncoder()
        else:

            # encoder_layer_config: 'regular_6',  'regular_4-sgdtv1_1-sgdt_1'
            encoder_layer_conf_list = parser_encoder_layers(encoder_layer_config)

            encoder_layer_list = []
            for l_type, num_l in encoder_layer_conf_list:
                assert l_type in SGDTEncoderLayerType and num_l > 0
                encoder_layer = SGDTEncoderLayerType[l_type](d_model, nhead, dim_feedforward,
                                                             dropout, activation, normalize_before,
                                                             )
                encoder_layer_list.append([encoder_layer, num_l])

            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = encoder_type(encoder_layer_list, encoder_norm)

        # --------------------------
        # Below are same with the original decoder.
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = decoder_type(decoder_layer, num_decoder_layers, decoder_norm,
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
                sgdt=None,
                mask_dict=None,  # mask_dict=mask_dict,
                class_embed=None,  # class_embed=class_embed
                **kwargs,
                ):
        """input:
            mask: not None,
            attn_mask: None,
        Returns:

        """
        # flatten NxCxHxW to HWxNxC # h= 25, w = 32
        bs, c, h, w = src.shape  # torch.Size([2, 256, 25, 32])
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # torch.Size([2, 256, 25, 32]) -> torch.Size([800, 2, 256])
        # refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)

        # -------------------------
        # sgdt_targets = self.sgdt.token_scoring_gt_generator .resize_sig_value_gt(
        #     sgdt_targets, feat_map_size=(h, w))
        # sgdt_targets = self.token_scoring_gt_generator(
        #     sgdt_target_raw, feat_map_size=(h, w), feat_map_mask=mask)
        # mask, (B, H, W)  -> (B, N), e.g., torch.Size([2, 30, 23]) -> torch.Size([2, 690])
        mask = mask.flatten(1)  # torch.Size([2, 21, 30]) -> torch.Size([2, 630]), src = torch.Size([630, 2, 256])

        # mask is used as src_key_padding_mask not mask even encoder has 'mask' input.
        memory, sgdt_output_list, pos_embed, mask, encoder_output_list = self.encoder(
            src, src_key_padding_mask=mask, pos=pos_embed,
            sgdt=sgdt,
        )

        # # if training token score only, we can skip the forward propagation of decoder.
        # if self.training and self.train_token_scoring_only:
        #     return None, None, sgdt_output_list

        if self.num_patterns > 0:
            l = tgt.shape[0]
            tgt[l - self.num_queries * self.num_patterns:] += \
                self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1)

        hs, references = self.decoder(tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
                                      pos=pos_embed, refpoints_unsigmoid=refpoint_embed)
        return hs, references, sgdt_output_list, encoder_output_list


class TransformerSGDT(Transformer):
    def __init__(self, d_model=512, nhead=8, num_queries=300, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_intermediate_dec=False, query_dim=4,
                 keep_query_pos=False, query_scale_type='cond_elewise', num_patterns=0, modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,

                 encoder_layer_config='regular_6',
                 align_encoder_decoder_layers_num=None,
                 ):
        super().__init__(d_model=d_model, nhead=nhead, num_queries=num_queries,
                         # num_encoder_layers=num_encoder_layers,
                         num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout,
                         activation=activation, normalize_before=normalize_before,
                         return_intermediate_dec=return_intermediate_dec, query_dim=query_dim,
                         keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                         num_patterns=num_patterns, modulate_hw_attn=modulate_hw_attn,
                         bbox_embed_diff_each_layer=bbox_embed_diff_each_layer,

                         encoder_layer_config=encoder_layer_config,
                         encoder_decoder_config='sgdt'
                         )

        assert align_encoder_decoder_layers_num is not None and isinstance(align_encoder_decoder_layers_num, int) and \
               align_encoder_decoder_layers_num > 0

        self.align_encoder_decoder_layers_num = align_encoder_decoder_layers_num

        # if align_encoder_decoder_layers_num = 2, then #0, 1, 2, 3
        # e0 -> e1 -> e2 -> e3 ->
        #       d0 -> d1 -> d2 -> d3 ->
        #                           e4 -> d4 -> e5 -> d5
        self.first_half_encoder_layer_ids = list(range(num_decoder_layers - align_encoder_decoder_layers_num))
        # 4, 5
        self.second_half_encoder_layer_ids = list(range(num_decoder_layers - align_encoder_decoder_layers_num,
                                                        num_decoder_layers))

    def forward(self, src, mask, refpoint_embed, pos_embed, tgt, attn_mask=None,
                sgdt=None,
                mask_dict=None,  # mask_dict=mask_dict,
                class_embed=None,  # class_embed=class_embed
                **kwargs,
                ):
        # flatten NxCxHxW to HWxNxC # h= 25, w = 32
        bs, c, h, w = src.shape  # torch.Size([2, 256, 25, 32])
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # torch.Size([2, 256, 25, 32]) -> torch.Size([800, 2, 256])
        # refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)

        mask = mask.flatten(1)  # torch.Size([2, 21, 30]) -> torch.Size([2, 630]), src = torch.Size([630, 2, 256])

        # do not accept the adapted pos_embed, mask from encoder as zeros pos_embed might be fed
        memory, sgdt_output_list, pos_embed, mask, encoder_output_list = self.encoder(
            src, src_key_padding_mask=mask, pos=pos_embed,
            encoder_layer_ids=self.first_half_encoder_layer_ids,
            sgdt=sgdt,
        )
        if self.encoder.norm is not None:
            memory = self.encoder.norm(memory)

        if self.num_patterns > 0:
            l = tgt.shape[0]
            tgt[l - self.num_queries * self.num_patterns:] += \
                self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1)

        # hs, references = self.decoder(tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
        #                               pos=pos_embed, refpoints_unsigmoid=refpoint_embed)
        hs, references, sgdt_output_list_decoder, encoder_output_list_decoder = self.decoder(
            tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
            pos=pos_embed, refpoints_unsigmoid=refpoint_embed,

            encoder=self.encoder,
            second_half_encoder_layer_ids=self.second_half_encoder_layer_ids,
            sgdt=sgdt,
            mask_dict=mask_dict,
            class_embed=class_embed,
        )
        sgdt_output_list += sgdt_output_list_decoder
        encoder_output_list += encoder_output_list_decoder
        return hs, references, sgdt_output_list, encoder_output_list


class TransformerSelfDistillation(Transformer):
    def __init__(self, d_model=512, nhead=8, num_queries=300, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_intermediate_dec=False, query_dim=4,
                 keep_query_pos=False, query_scale_type='cond_elewise', num_patterns=0, modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,
                 encoder_layer_config='regular_6',
                 align_encoder_decoder_layers_num=None,
                 ):
        super().__init__(d_model=d_model, nhead=nhead, num_queries=num_queries,
                         # num_encoder_layers=num_encoder_layers,
                         num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout,
                         activation=activation, normalize_before=normalize_before,
                         return_intermediate_dec=return_intermediate_dec, query_dim=query_dim,
                         keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                         num_patterns=num_patterns, modulate_hw_attn=modulate_hw_attn,
                         bbox_embed_diff_each_layer=bbox_embed_diff_each_layer,

                         encoder_layer_config=encoder_layer_config,
                         encoder_decoder_config='self_distillation'
                         )

        assert align_encoder_decoder_layers_num is not None and isinstance(align_encoder_decoder_layers_num, int) and \
               align_encoder_decoder_layers_num > 0

        self.align_encoder_decoder_layers_num = align_encoder_decoder_layers_num

        # if align_encoder_decoder_layers_num = 2, then #0, 1, 2, 3
        # e0 -> e1 -> e2 -> e3 ->
        #       d0 -> d1 -> d2 -> d3 ->
        #                           e4 -> d4 -> e5 -> d5
        self.student_encoder_layer_ids = list(
            range(num_decoder_layers - align_encoder_decoder_layers_num))

        # 4, 5
        self.teacher_encoder_layer_ids = list(
            range(num_decoder_layers - align_encoder_decoder_layers_num,
                  num_decoder_layers))

    def forward(self, src, mask, refpoint_embed, pos_embed, tgt, attn_mask=None,
                sgdt=None,
                mask_dict=None,  # mask_dict=mask_dict,
                class_embed=None,  # class_embed=class_embed
                **kwargs,
                ):
        # flatten NxCxHxW to HWxNxC # h= 25, w = 32
        bs, c, h, w = src.shape  # torch.Size([2, 256, 25, 32])
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # torch.Size([2, 256, 25, 32]) -> torch.Size([800, 2, 256])
        # refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)

        mask = mask.flatten(1)  # torch.Size([2, 21, 30]) -> torch.Size([2, 630]), src = torch.Size([630, 2, 256])

        # do not accept the adapted pos_embed, mask from encoder as zeros pos_embed might be fed
        memory, sgdt_output_list, pos_embed, mask, encoder_output_list = self.encoder(
            src, src_key_padding_mask=mask, pos=pos_embed,
            encoder_layer_ids=self.student_encoder_layer_ids,
            sgdt=sgdt,
        )
        if self.encoder.norm is not None:
            memory = self.encoder.norm(memory)

        if self.num_patterns > 0:
            l = tgt.shape[0]
            tgt[l - self.num_queries * self.num_patterns:] += \
                self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1)

        # hs, references = self.decoder(tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
        #                               pos=pos_embed, refpoints_unsigmoid=refpoint_embed)
        hs, references, sgdt_output_list_decoder, encoder_output_list_decoder = self.decoder(
            tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
            pos=pos_embed, refpoints_unsigmoid=refpoint_embed,

            encoder=self.encoder,
            encoder_output_list=encoder_output_list,
            second_half_encoder_layer_ids=self.teacher_encoder_layer_ids,
            sgdt=sgdt,
            mask_dict=mask_dict,
            class_embed=class_embed,
        )
        sgdt_output_list += sgdt_output_list_decoder
        encoder_output_list += encoder_output_list_decoder
        return hs, references, sgdt_output_list, encoder_output_list


class FeatureDistillationTransformerDecoder(TransformerDecoder):

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2

                encoder=None,
                encoder_output_list=None,
                second_half_encoder_layer_ids=None,
                sgdt=None,
                class_embed=None,
                mask_dict=None,
                ):
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        # import ipdb; ipdb.set_trace()
        sgdt_output_list = []
        encoder_output_list_decoder = []
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

            # --------------------- update the encoder features (memory), memory_key_padding_mask, pos here
            # len(self.layers) - 2
            # if align_encoder_decoder_layers_num = 2, then #0, 1, 2, 3
            # e0 -> e1 -> e2 -> e3 -> e4 ->
            #       d0 -> d1 -> d2 -> d3 -> d4 ->
            #                         e4 -> e5 -> d5
            encoder_layer_id = layer_id + 1
            if encoder is not None and isinstance(second_half_encoder_layer_ids, list) and \
                    encoder_layer_id in second_half_encoder_layer_ids:
                # modification: 1) never use proposal; 2) use the -2 encode layer instead of the
                # -1 encoder layer.
                memory, sgdt_output, pos, memory_key_padding_mask, encoder_output_list = encoder(
                    src=encoder_output_list[-2],     # memory,
                    src_key_padding_mask=memory_key_padding_mask,
                    pos=pos, sgdt=sgdt,
                    encoder_layer_ids=[encoder_layer_id],
                    # pos=pos if not encoder_without_pos else torch.zeros_like(pos),
                )
                sgdt_output_list += sgdt_output  # sgdt_output is a list with only only one item
                encoder_output_list_decoder += encoder_output_list
                # conduct norm if Transformer.normalize_before=True (output is not normalized),
                # encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
                if encoder.norm is not None:
                    memory = encoder.norm(memory)
            # --------------------------------------

        # all decoder layers are processed now.
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
                    sgdt_output_list,
                    encoder_output_list_decoder
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2),
                    sgdt_output_list,
                    encoder_output_list_decoder
                ]

        return output.unsqueeze(0), sgdt_output_list, encoder_output_list_decoder


class TransformerSGDTDecoder(TransformerDecoder):

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2

                encoder=None,
                second_half_encoder_layer_ids=None,
                sgdt=None,
                class_embed=None,
                mask_dict=None,
                ):
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        # import ipdb; ipdb.set_trace()
        sgdt_output_list = []
        encoder_output_list_decoder = []
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

            # --------------------- update the encoder features (memory), memory_key_padding_mask, pos here
            # len(self.layers) - 2
            # if align_encoder_decoder_layers_num = 2, then #0, 1, 2, 3
            # e0 -> e1 -> e2 -> e3 ->
            #       d0 -> d1 -> d2 -> d3 ->
            #                           e4 -> d4 -> e5 -> d5
            encoder_layer_id = layer_id + 1
            if encoder is not None and isinstance(second_half_encoder_layer_ids, list) and \
                    encoder_layer_id in second_half_encoder_layer_ids:

                if sgdt.gt_ratio is not None and sgdt.gt_ratio < 1.0:
                    assert sgdt.proposal_processor is not None
                    selected_proposals = self._extract_proposals(
                        intermediate=intermediate, ref_points=ref_points, reference_points=reference_points,
                        mask_dict=mask_dict, input_img_sizes=sgdt.get_input_img_sizes(),
                        proposal_processor=sgdt.proposal_processor,
                        class_embed=class_embed,
                    )
                    # update the proposal only, both the proposals in targets and sgdt_targets are updated.
                    sgdt.update_proposal_gt(selected_proposals=selected_proposals)

                # Below is how the decoder is called in Transformer:
                # hs, references = self.decoder(tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
                #                               pos=pos_embed, refpoints_unsigmoid=refpoint_embed)
                memory, sgdt_output, pos, memory_key_padding_mask, encoder_output_list = encoder(
                    src=memory, src_key_padding_mask=memory_key_padding_mask,
                    pos=pos, sgdt=sgdt,
                    encoder_layer_ids=[encoder_layer_id],
                    # pos=pos if not encoder_without_pos else torch.zeros_like(pos),
                )
                sgdt_output_list += sgdt_output  # sgdt_output is a list with only only one item
                encoder_output_list_decoder += encoder_output_list
                # conduct norm if Transformer.normalize_before=True (output is not normalized),
                # encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
                if encoder.norm is not None:
                    memory = encoder.norm(memory)
            # --------------------------------------

        # all decoder layers are processed now.
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
                    sgdt_output_list,
                    encoder_output_list_decoder
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2),
                    sgdt_output_list,
                    encoder_output_list_decoder
                ]

        return output.unsqueeze(0), sgdt_output_list, encoder_output_list_decoder

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


def build_transformer(args):
    if args.feature_distillation:
        transformer = TransformerSelfDistillation
    elif args.align_encoder_decoder_layers_num > 0:
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
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=4,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,

        # --------------------
        # num_encoder_sgdt_layers=args.num_encoder_sgdt_layers,
        # encoder_without_pos=args.encoder_without_pos,
        # encoder_sgdt_layer_version=args.encoder_sgdt_layer_version,
        # reclaim_padded_region=args.reclaim_padded_region,
        # token_scoring_discard_split_criterion=args.token_scoring_discard_split_criterion,
        encoder_layer_config=args.encoder_layer_config,
        align_encoder_decoder_layers_num=args.align_encoder_decoder_layers_num,
        # train_token_scoring_only=args.train_token_scoring_only
    )
