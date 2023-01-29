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


# import math
# import copy
# import os
from typing import Optional, List
from util.misc import inverse_sigmoid

import torch
import torch.nn.functional as F
from torch import nn, Tensor

# ------------------- TTI Modification

# the attention in TransformerDecoderLayer is memory consuming, so used the latest pytorch implementation.
from models.DN_DAB_DETR.transformer import MLP, TransformerDecoderLayer, \
    _get_activation_fn, _get_clones, gen_sineembed_for_position
from models.DN_DAB_DETR.attention import MultiheadAttention as DecoderMultiheadAttention
from models.sgdt.attn import MultiheadAttention
from models.sgdt.attn_extended import MultiheadAttentionExtended
from models.sgdt.attn_share_QV import MultiheadAttentionShareQV
from models.sgdt.attn_share_V import MultiheadAttentionShareV
# from torch.nn import MultiheadAttention

from models.sgdt.sgdt_module import SGDT_module, get_valid_token_mask, TokenScoringConfigParser
# from models.sgdt.scoring_gt import resize_sgdt_target
from models.DN_DAB_DETR.dn_components import dn_post_process
from models.sgdt.scoring_gt import update_targets_with_proposals
from models.sgdt.sgdt_components import parser_encoder_decoder_layers


# from models.sgdt.sgdt_ import SGDT


def extract_adapted_token_pos_embed(adapted_token_dict, pos: Optional[Tensor]):
    """
    return extracted pos based on tokens_small_obj, tokens_to_discard in adapted_token_dict
    Args:
        adapted_token_dict: a dict included tokens_small_obj, tokens_to_discard
        pos:  position_embedding, (N, B, C), e.g., torch.Size([800, 2, 256])
            requires_grad = False (as sine encoding is used, not learnable position_embedding)
    Returns:
    """
    return pos


def mark_encoder_feature_by_fg_gt(memory, sgdt):
    # set the last feature dimension to be the ft_gt mask
    assert isinstance(sgdt.sgdt_targets, dict) and 'fg_gt' in sgdt.sgdt_targets
    # memory: N, B, C torch.Size([756, 2, 256]);  sgdt['fg_gt']: N, B shape torch.Size([756, 2])
    memory[:, :, -1] = memory[:, :, -1] * 0 + sgdt.sgdt_targets['fg_gt'].type(memory.dtype)
    return memory


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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        #
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

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

        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]

        # need_weights=True, average_attn_weights=False
        # attn_output_weight_logits: (b, num_heads, N, N),torch.Size([2,8, 888, 888]) (bsz, num_heads, tgt_len, src_len)
        src2, _, attn_output_weight_logits = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                                            key_padding_mask=src_key_padding_mask,
                                                            need_weights=True, average_attn_weights=False)  #

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_output_weight_logits


class TransformerEncoderMarkFgBgLayer(TransformerEncoderLayer):
    # Q from adapted tokens, k, v from original token

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt=None,
                ):
        # ---------------
        marking_encoder_layer_fg1_bg0 = None
        if sgdt is not None and sgdt.marking_encoder_layer_fg1_bg0:
            marking_encoder_layer_fg1_bg0 = sgdt.marking_encoder_layer_fg1_bg0

        q = self.with_pos_embed(src, pos)
        # ---------------
        k, v = q.clone(), src.clone()
        if marking_encoder_layer_fg1_bg0 == 'K':
            k = mark_encoder_feature_by_fg_gt(k, sgdt)
        elif marking_encoder_layer_fg1_bg0 == 'V':
            v = mark_encoder_feature_by_fg_gt(v, sgdt)
        elif marking_encoder_layer_fg1_bg0 == 'Q':
            q = mark_encoder_feature_by_fg_gt(q, sgdt)

        src2, _, attn_output_weight_logits = self.self_attn(q, k, value=v, attn_mask=src_mask,
                                                            key_padding_mask=src_key_padding_mask,
                                                            need_weights=True, average_attn_weights=False)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # ---------------
        if marking_encoder_layer_fg1_bg0 == 'FFN_Out':
            mark_encoder_feature_by_fg_gt(src, sgdt)

        return src, attn_output_weight_logits


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

        # q_adapted_pos = extract_adapted_token_pos_embed(adapted_token_dict=sgdt_output, pos=pos)
        q_adapted_pos = pos
        q = self.with_pos_embed(sgdt_output['x'], pos=q_adapted_pos)  # using adapted tokens as queries

        # # tcformer_layers.py forward(self, q_dict, kv_dict): so k v should come from the same src.
        # #  but here we q, v are from the same src, because the attention means the attention to the key
        # # Do not update src_key_padding_mask here, because k, v tokens are not adapted, adaption of q tokens will
        # # will not affect the key_padding_mask.
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]

        src2, _, attn_output_weight_logits = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                                            key_padding_mask=src_key_padding_mask,
                                                            need_weights=True, average_attn_weights=False)  #

        # check if I need to use 'src' or sgdt_output['x'] (maybe TCformer used sgdt_output['x']?),
        # confirmed: TCformer used sgdt_output['x'] as the short path, i.e., use down_dict as src from
        # (down_dict, token_dict)
        # If use 'src' same as in SGDTV1, then the pos of the the tokens in 'src' and 'self.dropout1(src2)
        # are not consistent due the token adaption, so I should use sgdt_output['x']
        src = sgdt_output['x'] + self.dropout1(src2)

        # TODO: I should use [based on the identity mapping]:
        #  src = src + self.dropout1(src2)

        src = self.norm1(src)
        # TODO: replace with dwconv as in TCFormer
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # Pass the related infor of sgdt module for loss calculation and pos adaption.
        # The input pos is the scaled pos, not the original pos, so we do not pass q_adapted_pos
        # out, but just return a flag saying that the original pos should be adapted outside this
        # function with the original pos..
        # sgdt_output.update(dict(adapted_pos=True))
        sgdt_output.update(dict(adapted_pos=False))

        # src_key_padding_mask can be safely updated here.
        # ** ATTENTION **, if this section is put in Transformer, we should update 'mask',
        # i.e., mask = sgdt_output['src_mask_reclaimed']
        # but if it is inside TransformerEncoder, we should update the 'src_key_padding_mask'.
        assert 'src_mask_reclaimed' in sgdt_output and sgdt_output['src_mask_reclaimed'] is not None
        src_key_padding_mask = sgdt_output['src_mask_reclaimed']

        return src, attn_output_weight_logits, sgdt_output, src_key_padding_mask


class TransformerEncoderSGDTLayerV0NoMaskUpdate(TransformerEncoderSGDTLayer):
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
        # do not update src_key_padding_mask in this version
        src, attn_output_weight_logits, sgdt_output, _ = super().forward(
            src=src, src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask, pos=pos, sgdt=sgdt
        )
        return src, attn_output_weight_logits, sgdt_output, src_key_padding_mask


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
        # k_adapted_pos = extract_adapted_token_pos_embed(
        #     adapted_token_dict=sgdt_output, pos=pos)
        k_adapted_pos = pos
        k = self.with_pos_embed(sgdt_output['x'], pos=k_adapted_pos)

        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]

        # 1. value is not src anymore, as k has been updated.
        # 2. some tokens are reclaimed so they are not invalid tokens any more, we need to
        # adapt the invalid token mask as well.
        assert 'src_mask_reclaimed' in sgdt_output and sgdt_output['src_mask_reclaimed'] is not None
        # src2 = self.self_attn(q, k, value=sgdt_output['x'], attn_mask=src_mask,
        #                       key_padding_mask=sgdt_output['src_mask_reclaimed'])[0]

        src2, _, attn_output_weight_logits = self.self_attn(q, k, value=sgdt_output['x'], attn_mask=src_mask,
                                                            key_padding_mask=sgdt_output['src_mask_reclaimed'],
                                                            need_weights=True, average_attn_weights=False)  #

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

        return src, attn_output_weight_logits, sgdt_output, src_key_padding_mask


class TransformerEncoderSGDTLayerUpdateK(TransformerEncoderSGDTLayerV1):
    # Q from original token, k, v from adapted tokens.

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt=None,
                ):
        q = self.with_pos_embed(src, pos)  # k, v are from the original tokens.
        sgdt_output = sgdt(x=src, mask=src_key_padding_mask)
        k_adapted_pos = pos
        k = self.with_pos_embed(sgdt_output['x'], pos=k_adapted_pos)
        assert 'src_mask_reclaimed' in sgdt_output and sgdt_output['src_mask_reclaimed'] is not None

        # ################################################ only modification here
        # not update k
        # src2, _, attn_output_weight_logits = self.self_attn(q, k, value=sgdt_output['x'], attn_mask=src_mask,
        #                                            key_padding_mask=sgdt_output['src_mask_reclaimed'],
        #                                            need_weights=True, average_attn_weights=False)  #

        src2, _, attn_output_weight_logits = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                                            key_padding_mask=sgdt_output['src_mask_reclaimed'],
                                                            need_weights=True, average_attn_weights=False)
        # ################################################

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # k_adapted_pos is not used for self-attention inside the encoder layer,
        # should not be used outside Encoder layer
        sgdt_output.update(dict(adapted_pos=False))

        return src, attn_output_weight_logits, sgdt_output, src_key_padding_mask


class TransformerEncoderSGDTLayerUpdateV(TransformerEncoderSGDTLayerV1):
    # Q from original token, k, v from adapted tokens.

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt=None,
                ):
        # ################################################   not update q, k, only update v.
        q = k = self.with_pos_embed(src, pos)
        sgdt_output = sgdt(x=src, mask=src_key_padding_mask)
        # k_adapted_pos = pos
        # k = self.with_pos_embed(sgdt_output['x'], pos=k_adapted_pos)
        # assert 'src_mask_reclaimed' in sgdt_output and sgdt_output['src_mask_reclaimed'] is not None

        # src2, _, attn_output_weight_logits = self.self_attn(q, k, value=sgdt_output['x'], attn_mask=src_mask,
        #                                            key_padding_mask=sgdt_output['src_mask_reclaimed'],
        #                                            need_weights=True, average_attn_weights=False)  #

        src2, _, attn_output_weight_logits = self.self_attn(q, k, value=sgdt_output['x'], attn_mask=src_mask,
                                                            key_padding_mask=src_key_padding_mask,
                                                            need_weights=True, average_attn_weights=False)
        # ################################################

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # k_adapted_pos is not used for self-attention inside the encoder layer,
        # should not be used outside Encoder layer
        sgdt_output.update(dict(adapted_pos=False))

        return src, attn_output_weight_logits, sgdt_output, src_key_padding_mask


class TransformerEncoderSGDTLayerUpdateMHAOut(TransformerEncoderSGDTLayerV1):
    # Q from original token, k, v from adapted tokens.

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt=None,
                ):
        q = k = self.with_pos_embed(src, pos)
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]

        src2, _, attn_output_weight_logits = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                                            key_padding_mask=src_key_padding_mask,
                                                            need_weights=True, average_attn_weights=False)  #

        # =================-
        # update the MHA-out
        sgdt_output = sgdt(x=src2, mask=src_key_padding_mask)
        src2 = sgdt_output['x']
        # =================-

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # # k_adapted_pos is not used for self-attention inside the encoder layer,
        # # should not be used outside Encoder layer
        sgdt_output.update(dict(adapted_pos=False))

        return src, attn_output_weight_logits, sgdt_output, src_key_padding_mask


class TransformerEncoderSGDTLayerUpdateMHAFeature(TransformerEncoderSGDTLayerV1):
    # Q from original token, k, v from adapted tokens.

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt=None,
                ):
        q = k = self.with_pos_embed(src, pos)
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]

        src2, _, attn_output_weight_logits = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                                            key_padding_mask=src_key_padding_mask,
                                                            need_weights=True, average_attn_weights=False)  #

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # =================-
        # # update the MHA-out
        sgdt_output = sgdt(x=src, mask=src_key_padding_mask)
        src = sgdt_output['x']
        # =================-

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # # k_adapted_pos is not used for self-attention inside the encoder layer,
        # # should not be used outside Encoder layer
        sgdt_output.update(dict(adapted_pos=False))

        return src, attn_output_weight_logits, sgdt_output, src_key_padding_mask


class TransformerEncoderSGDTLayerUpdateFFNOut(TransformerEncoderSGDTLayerV1):
    # Q from original token, k, v from adapted tokens.

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt=None,
                ):
        q = k = self.with_pos_embed(src, pos)
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]

        src2, _, attn_output_weight_logits = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                                            key_padding_mask=src_key_padding_mask,
                                                            need_weights=True, average_attn_weights=False)  #

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        # =================-
        # # update the FFN-out
        sgdt_output = sgdt(x=src2, mask=src_key_padding_mask)
        src2 = sgdt_output['x']
        # =================-

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # # k_adapted_pos is not used for self-attention inside the encoder layer,
        # # should not be used outside Encoder layer
        sgdt_output.update(dict(adapted_pos=False))

        return src, attn_output_weight_logits, sgdt_output, src_key_padding_mask


class TransformerEncoderSGDTLayerUpdateFFNFeature(TransformerEncoderSGDTLayerV1):
    # Q from original token, k, v from adapted tokens.

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt=None,
                ):
        q = k = self.with_pos_embed(src, pos)
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]

        src2, _, attn_output_weight_logits = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                                            key_padding_mask=src_key_padding_mask,
                                                            need_weights=True, average_attn_weights=False)  #

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # =================-
        # # update the FFN-features
        sgdt_output = sgdt(x=src, mask=src_key_padding_mask)
        src = sgdt_output['x']
        # =================-

        # # k_adapted_pos is not used for self-attention inside the encoder layer,
        # # should not be used outside Encoder layer
        sgdt_output.update(dict(adapted_pos=False))

        return src, attn_output_weight_logits, sgdt_output, src_key_padding_mask


class TransformerEncoderSGDTLayerUpdateQKV(TransformerEncoderSGDTLayerV1):
    # Q from original token, k, v from adapted tokens.

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt=None,
                ):
        # =================-
        # # update the input
        sgdt_output = sgdt(x=src, mask=src_key_padding_mask)
        src = sgdt_output['x']
        # =================-

        q = k = self.with_pos_embed(src, pos)
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]

        src2, _, attn_output_weight_logits = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                                            key_padding_mask=src_key_padding_mask,
                                                            need_weights=True, average_attn_weights=False)  #

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # # k_adapted_pos is not used for self-attention inside the encoder layer,
        # # should not be used outside Encoder layer
        sgdt_output.update(dict(adapted_pos=False))

        return src, attn_output_weight_logits, sgdt_output, src_key_padding_mask


class TransformerEncoderSharedSelfAttnLayer(TransformerEncoderLayer):
    # used pre_calculated_attn from outside

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                         dropout=dropout, activation=activation, normalize_before=normalize_before)

        # update the self_attn
        del self.self_attn
        self.self_attn = MultiheadAttentionExtended(d_model, nhead, dropout=dropout)

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,

                pre_calculated_attn: Optional[Tensor] = None,
                ):
        q = k = self.with_pos_embed(src, pos)

        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]

        #  need_weights=True, average_attn_weights=False
        # attn_output_weight_logits : (b, num_heads, N, N)  torch.Size([2,8, 888, 888]) (bsz, num_heads, tgt_len, src_len)
        src2, _, attn_output_weight_logits = self.self_attn(
            q, k, value=src, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True, average_attn_weights=False,
            # use pre_calculated_attn
            pre_calculated_attn=pre_calculated_attn,  # torch.Size([2, 8, 696, 696])
            freeze_wq=True,
            freeze_wk=True,
        )

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_output_weight_logits


class TransformerEncoderSGDTLayerUpdateKExtendedSelfAttn(TransformerEncoderSharedSelfAttnLayer):
    # q, v from original token, k from adapted tokens.
    # Old name: TransformerEncoderSGDTLayerV1ExtendedSelfAttn

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
        # k_adapted_pos = extract_adapted_token_pos_embed(
        #     adapted_token_dict=sgdt_output, pos=pos)
        k_adapted_pos = pos
        k = self.with_pos_embed(sgdt_output['x'], pos=k_adapted_pos)

        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]

        # 1. value is not src anymore, as k has been updated.
        # 2. some tokens are reclaimed so they are not invalid tokens any more, we need to
        # adapt the invalid token mask as well.
        assert 'src_mask_reclaimed' in sgdt_output and sgdt_output['src_mask_reclaimed'] is not None
        # src2 = self.self_attn(q, k, value=sgdt_output['x'], attn_mask=src_mask,
        #                       key_padding_mask=sgdt_output['src_mask_reclaimed'])[0]

        # src2, _, attn_output_weight_logits = self.self_attn(q, k, value=sgdt_output['x'], attn_mask=src_mask,
        #                                            key_padding_mask=sgdt_output['src_mask_reclaimed'],
        #                                            need_weights=True, average_attn_weights=False)  #

        src2, _, attn_output_weight_logits = self.self_attn(
            q, k, value=src, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True, average_attn_weights=False,
            # freeze the softmax.
            freeze_wq=True,
            freeze_wk=True,
        )

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

        return src, attn_output_weight_logits, sgdt_output, src_key_padding_mask


class TransformerEncoderLayerSTMarkFgBgShareQVFreezeAllExceptWK(TransformerEncoderLayer):
    #

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                         dropout=dropout, activation=activation, normalize_before=normalize_before)

        # update the self_attn
        del self.self_attn
        # self.self_attn = MultiheadAttentionExtended(d_model, nhead, dropout=dropout)

        self.self_attn = MultiheadAttentionShareQV(d_model, nhead, dropout=dropout)

        self.teacher_encoder_layer = _TransformerEncoderLayerFFNOnly(
            d_model=d_model, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation,
            normalize_before=normalize_before)

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt=None,
                # pre_calculated_attn: Optional[Tensor] = None,
                ):
        q = k = self.with_pos_embed(src, pos)

        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]

        #  need_weights=True, average_attn_weights=False
        # attn_output_weight_logits : (b, num_heads, N, N)  torch.Size([2,8, 888, 888]) (bsz, num_heads, tgt_len, src_len)
        # no backpropagation for X
        k_teacher = mark_encoder_feature_by_fg_gt(k.clone().detach(), sgdt)
        src2_s_t_tuple, _, attn_output_weight_logits_s_t_tuple = self.self_attn(
            q, k, value=src, key_teacher=k_teacher,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True, average_attn_weights=False,
            # use pre_calculated_attn
            # pre_calculated_attn=pre_calculated_attn,  # torch.Size([2, 8, 696, 696])
            # freeze_wq=True,
            # freeze_wk=True,
        )
        src2, src2_teacher = src2_s_t_tuple
        attn_map_logits, attn_output_weight_logits_teacher = attn_output_weight_logits_s_t_tuple
        src_teacher = self.teacher_encoder_layer(
            src=src.detach().clone(),
            mha_out=src2_teacher,  # mha_out
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # return (src, src_teacher), attn_output_weight_logits_s_t_tuple
        return src, attn_map_logits, src_teacher, attn_output_weight_logits_teacher


class TransformerEncoderLayerSTMarkFgBgShareVNoFreeze(TransformerEncoderLayer):
    #

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                         dropout=dropout, activation=activation, normalize_before=normalize_before)

        # update the self_attn
        del self.self_attn
        # self.self_attn = MultiheadAttentionExtended(d_model, nhead, dropout=dropout)

        self.self_attn = MultiheadAttentionShareV(d_model, nhead, dropout=dropout)

        self.teacher_encoder_layer = _TransformerEncoderLayerFFNOnly(
            d_model=d_model, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation,
            normalize_before=normalize_before)

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt=None,
                # pre_calculated_attn: Optional[Tensor] = None,
                ):
        q = k = self.with_pos_embed(src, pos)

        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]

        #  need_weights=True, average_attn_weights=False
        # attn_output_weight_logits : (b, num_heads, N, N)  torch.Size([2,8, 888, 888]) (bsz, num_heads, tgt_len, src_len)
        # no backpropagation for X
        k_teacher = mark_encoder_feature_by_fg_gt(k.clone().detach(), sgdt)
        src2_s_t_tuple, _, attn_output_weight_logits_s_t_tuple = self.self_attn(
            q, k, value=src, query_teacher=q.clone(), key_teacher=k_teacher,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True, average_attn_weights=False,
            # use pre_calculated_attn
            # pre_calculated_attn=pre_calculated_attn,  # torch.Size([2, 8, 696, 696])
            # freeze_wq=True,
            # freeze_wk=True,
        )
        src2, src2_teacher = src2_s_t_tuple
        attn_map_logits, attn_output_weight_logits_teacher = attn_output_weight_logits_s_t_tuple
        src_teacher = self.teacher_encoder_layer(
            src=src,  # src.detach().clone(),
            mha_out=src2_teacher,  # mha_out
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # return (src, src_teacher), attn_output_weight_logits_s_t_tuple
        return src, attn_map_logits, src_teacher, attn_output_weight_logits_teacher


class TransformerEncoderLayerSTSGDTShareVNoFreeze(TransformerEncoderLayerSTMarkFgBgShareVNoFreeze):

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt=None,
                # pre_calculated_attn: Optional[Tensor] = None,
                ):
        q = k = self.with_pos_embed(src, pos)

        # ============================
        # k_teacher = mark_encoder_feature_by_fg_gt(k.clone().detach(), sgdt)
        sgdt_output = sgdt(x=src, mask=src_key_padding_mask, )
        k_adapted_pos = pos
        k_teacher = self.with_pos_embed(sgdt_output['x'], pos=k_adapted_pos)
        # ============================

        src2_s_t_tuple, _, attn_output_weight_logits_s_t_tuple = self.self_attn(
            q, k, value=src, query_teacher=q.clone(), key_teacher=k_teacher,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True, average_attn_weights=False,
            # use pre_calculated_attn
            # pre_calculated_attn=pre_calculated_attn,  # torch.Size([2, 8, 696, 696])
            # freeze_wq=True,
            # freeze_wk=True,
        )
        src2, src2_teacher = src2_s_t_tuple
        attn_map_logits, attn_output_weight_logits_teacher = attn_output_weight_logits_s_t_tuple

        src_teacher = self.teacher_encoder_layer(
            src=src,  # src.detach().clone(),
            mha_out=src2_teacher,  # mha_out
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # return (src, src_teacher), attn_output_weight_logits_s_t_tuple
        return src, attn_map_logits, src_teacher, attn_output_weight_logits_teacher


class TransformerEncoderLayerSTShareSGDTUpdateKAttn(TransformerEncoderLayer):
    #

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                         dropout=dropout, activation=activation, normalize_before=normalize_before)

        self.student_encoder_layer = TransformerEncoderSharedSelfAttnLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation,
            normalize_before=normalize_before)

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sgdt=None,
                # pre_calculated_attn: Optional[Tensor] = None,
                ):
        q = self.with_pos_embed(src, pos)  # k, v are from the original tokens.
        sgdt_output = sgdt(x=src, mask=src_key_padding_mask)
        k_adapted_pos = pos
        k = self.with_pos_embed(sgdt_output['x'], pos=k_adapted_pos)
        assert 'src_mask_reclaimed' in sgdt_output and sgdt_output['src_mask_reclaimed'] is not None
        src2, _, attn_output_weight_logits = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                                            key_padding_mask=sgdt_output['src_mask_reclaimed'],
                                                            need_weights=True, average_attn_weights=False)
        # the student branch is not allowed to update the attn by backpropagation
        pre_calculated_attn = F.softmax(attn_output_weight_logits.clone().detach(), dim=-1)
        src_student, _ = self.student_encoder_layer(
            src=src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, pos=pos,

            # use pre_calculated_attn
            pre_calculated_attn=pre_calculated_attn,
            # use pre_calculated_attn means w_q, w_k will have zero gradients,
            # so we do no need to set freeze_wq, freeze_wk

        )

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # return src, attn_map_logits, src_teacher, attn_output_weight_logits_teacher
        return src_student, attn_output_weight_logits, src, attn_output_weight_logits


class _TransformerEncoderLayerFFNOnly(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # ====================
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # ====================

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

    def forward(self,
                src,
                mha_out,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # q = k = self.with_pos_embed(src, pos)

        # src2, _, attn_output_weight_logits = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                                                     key_padding_mask=src_key_padding_mask,
        #                                                     need_weights=True, average_attn_weights=False)  #
        src2 = mha_out
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


SGDTEncoderLayerType = {
    'regular': TransformerEncoderLayer,
    'sgdt': TransformerEncoderSGDTLayer,
    'sgdtv0': TransformerEncoderSGDTLayerV0NoMaskUpdate,
    'sgdtv1': TransformerEncoderSGDTLayerV1,
    'sgdt+k': TransformerEncoderSGDTLayerUpdateK,
    'sgdt+v': TransformerEncoderSGDTLayerUpdateV,

    'sgdt+qkv': TransformerEncoderSGDTLayerUpdateQKV,
    'sgdt+mha+out': TransformerEncoderSGDTLayerUpdateMHAOut,
    'sgdt+mha+feature': TransformerEncoderSGDTLayerUpdateMHAFeature,
    'sgdt+ffn+out': TransformerEncoderSGDTLayerUpdateFFNOut,
    'sgdt+ffn+feature': TransformerEncoderSGDTLayerUpdateFFNFeature,

    'sgdtSharedAttn': TransformerEncoderSharedSelfAttnLayer,
    'sgdtv1FreezeSelfAttn': TransformerEncoderSGDTLayerUpdateKExtendedSelfAttn,
    'sgdtMarkFgBg': TransformerEncoderMarkFgBgLayer,

    # teacher student both in a single encoder layer
    'parallelSTECMarkFgBgVFreezeAll': TransformerEncoderLayerSTMarkFgBgShareQVFreezeAllExceptWK,
    'parallelSTECMarkFgBgShareVNoFreeze': TransformerEncoderLayerSTMarkFgBgShareVNoFreeze,
    'parallelSTECShareSGDTUpdateKAttn': TransformerEncoderLayerSTShareSGDTUpdateKAttn,
    'parallelSTECSGDTShareVNoFreeze': TransformerEncoderLayerSTSGDTShareVNoFreeze,
}


class TransformerEncoder(nn.Module):
    # process all encoder layers in one pass

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

        self.catch_encoder_output_from_layer_id = 4

    def forward_subset_encoder_layers(self, src,
                                      mask: Optional[Tensor] = None,
                                      src_key_padding_mask: Optional[Tensor] = None,
                                      pos: Optional[Tensor] = None,

                                      encoder_layer_ids=None,
                                      sgdt=None,
                                      teacher_encoder_output_list=None,
                                      ):
        # mask is always None, all others are not None.
        # the src_mask is only used by src_key_padding_mask (not mask)
        output = src  # torch.Size([800, 2, 256])

        if encoder_layer_ids is None:
            encoder_layers = self.layers
        else:
            assert isinstance(encoder_layer_ids, list) and len(encoder_layer_ids) <= len(self.layers)
            encoder_layers = [self.layers[k] for k in encoder_layer_ids]

        sgdt_layer_types = [v for k, v in SGDTEncoderLayerType.items()]

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
            output_online_teacher, attn_map_online_teacher = None, None
            if type(layer).__name__ == 'TransformerEncoderLayer':  # Regular TransformerEncoderLayer
                output, attn_map_logits = layer(output, src_mask=mask,
                                                src_key_padding_mask=src_key_padding_mask,
                                                pos=pos * pos_scales
                                                )

            elif type(layer).__name__ == 'TransformerEncoderSharedSelfAttnLayer':
                assert teacher_encoder_output_list is not None and len(teacher_encoder_output_list) > layer_id and \
                       'attn_map_logits' in teacher_encoder_output_list[layer_id]

                pre_calculated_attn = F.softmax(teacher_encoder_output_list[layer_id]['attn_map_logits'], dim=-1)
                output, attn_map_logits = layer(
                    output, src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos=pos * pos_scales,  # scaled pos is passed in
                    pre_calculated_attn=pre_calculated_attn,
                )
            elif type(layer).__name__ == 'TransformerEncoderMarkFgBgLayer':
                output, attn_map_logits = layer(
                    output, src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos=pos * pos_scales,  # scaled pos is passed in
                    sgdt=sgdt,
                )
            elif type(layer).__name__ in ['TransformerEncoderLayerSTMarkFgBgShareQVFreezeAllExceptWK',
                                          'TransformerEncoderLayerSTMarkFgBgShareVNoFreeze',
                                          'TransformerEncoderLayerSTShareSGDTUpdateKAttn',
                                          'TransformerEncoderLayerSTSGDTShareVNoFreeze',
                                          ]:

                output, attn_map_logits, output_online_teacher, attn_map_online_teacher = layer(
                    output, src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos=pos * pos_scales,  # scaled pos is passed in
                    sgdt=sgdt,
                )
            elif type(layer) in sgdt_layer_types:
                # ['TransformerEncoderSGDTLayer', 'TransformerEncoderSGDTLayerV1']
                # type(layer).__name__ in sgdt_layer_types
                # SGDT layers (v0, v1) used for token adaption

                output, attn_map_logits, sgdt_output, src_key_padding_mask = layer(
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
            else:
                raise NotImplementedError
                # -------------------
            # return a dict to be more flexible  ['attn_map_logits']
            if output_online_teacher is None and attn_map_online_teacher is None:
                encoder_output_list.append(dict(
                    feat=output if layer_id >= self.catch_encoder_output_from_layer_id else None,
                    attn_map_logits=attn_map_logits if layer_id >= self.catch_encoder_output_from_layer_id else None,
                    )
                )
            else:
                encoder_output_list.append(dict(
                    feat=output if layer_id >= self.catch_encoder_output_from_layer_id else None,
                    attn_map_logits=attn_map_logits if layer_id >= self.catch_encoder_output_from_layer_id else None,
                    output_online_teacher=output_online_teacher,
                    attn_map_online_teacher=attn_map_online_teacher,
                    )
                )
            # encoder_output_list.append(output)
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
                teacher_encoder_output_list=None
                ):
        # process all encoder layers, so encoder_layer_ids=None
        output, sgdt_output_list, pos, src_key_padding_mask, encoder_output_list = self.forward_subset_encoder_layers(
            src=src, mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos, sgdt=sgdt,
            teacher_encoder_output_list=teacher_encoder_output_list,
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
    # # process a subset of encoder layers each time.

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


class TransformerDecoderLayerSTMarkFgBgShareQV(nn.Module): # TODO: to continue the modification
    # Two decoder layers with shared self-attn, and Q, V in the cross-attention.
    # FFN is not shared.

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

            # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)
            self.self_attn = DecoderMultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)

        # self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
        # self.cross_attn = MultiheadAttentionShareQV(
        #     d_model * 2, nhead, dropout=dropout, vdim=d_model
        # )

        self.cross_attn = DecoderMultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
        # out_proj is defined
        self.cross_attn_teacher = DecoderMultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model

        self.ffn = _DecoderFFN(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout,
                               activation=activation)
        self.ffn_teacher = _DecoderFFN(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout,
                                       activation=activation)

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
                is_first=False,

                sgdt=None,
                ):

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

        # k_teacher = mark_encoder_feature_by_fg_gt(k.clone().detach(), sgdt)
        k_teacher = mark_encoder_feature_by_fg_gt(k, sgdt)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        # self.cross_attn_teacher
        tgt2_s_t_tuple, _, attn_output_weight_logits_s_t_tuple = self.cross_attn(
            query=q, key=k, value=v,
            key_teacher=k_teacher,  # ##
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,

            freeze_q=False,
            freeze_v=False,
            need_weights=True, average_attn_weights=False,  # ##
        )
        # ========== End of Cross-Attention =============

        tgt2, tgt2_teacher = tgt2_s_t_tuple
        attn_map_logits, attn_output_weight_logits_teacher = attn_output_weight_logits_s_t_tuple

        tgt = self.ffn(tgt=tgt, tgt2=tgt2)
        tgt_teacher = self.ffn_teacher(tgt=tgt, tgt2=tgt2_teacher)

        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)
        # return tgt
        return (tgt, attn_map_logits, tgt_teacher, attn_output_weight_logits_teacher)


class _DecoderFFN(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        # # Decoder Self-Attention
        # if not rm_self_attn_decoder:
        #     self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        #     self.sa_qpos_proj = nn.Linear(d_model, d_model)
        #     self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        #     self.sa_kpos_proj = nn.Linear(d_model, d_model)
        #     self.sa_v_proj = nn.Linear(d_model, d_model)
        #     self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)
        #
        #     self.norm1 = nn.LayerNorm(d_model)
        #     self.dropout1 = nn.Dropout(dropout)
        #
        # # Decoder Cross-Attention
        # self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        # self.ca_qpos_proj = nn.Linear(d_model, d_model)
        # self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        # self.ca_kpos_proj = nn.Linear(d_model, d_model)
        # self.ca_v_proj = nn.Linear(d_model, d_model)
        # self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        # self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

        # self.nhead = nhead
        # self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        # self.normalize_before = normalize_before
        # self.keep_query_pos = keep_query_pos

    def forward(self, tgt, tgt2):
        # # ========== Begin of Self-Attention =============
        # if not self.rm_self_attn_decoder:
        #     # Apply projections here
        #     # shape: num_queries x batch_size x 256
        #     q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        #     q_pos = self.sa_qpos_proj(query_pos)
        #     k_content = self.sa_kcontent_proj(tgt)
        #     k_pos = self.sa_kpos_proj(query_pos)
        #     v = self.sa_v_proj(tgt)
        #
        #     num_queries, bs, n_model = q_content.shape
        #     hw, _, _ = k_content.shape
        #
        #     q = q_content + q_pos
        #     k = k_content + k_pos
        #
        #     tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
        #                         key_padding_mask=tgt_key_padding_mask)[0]
        #     # ========== End of Self-Attention =============
        #
        #     tgt = tgt + self.dropout1(tgt2)
        #     tgt = self.norm1(tgt)
        #
        # # ========== Begin of Cross-Attention =============
        # # Apply projections here
        # # shape: num_queries x batch_size x 256
        # q_content = self.ca_qcontent_proj(tgt)
        # k_content = self.ca_kcontent_proj(memory)
        # v = self.ca_v_proj(memory)
        #
        # num_queries, bs, n_model = q_content.shape
        # hw, _, _ = k_content.shape
        #
        # k_pos = self.ca_kpos_proj(pos)
        #
        # # For the first decoder layer, we concatenate the positional embedding predicted from
        # # the object query (the positional embedding) into the original query (key) in DETR.
        # if is_first or self.keep_query_pos:
        #     q_pos = self.ca_qpos_proj(query_pos)
        #     q = q_content + q_pos
        #     k = k_content + k_pos
        # else:
        #     q = q_content
        #     k = k_content
        #
        # q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        # query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        # query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        # q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        # k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        # k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        # k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)
        #
        # tgt2 = self.cross_attn(query=q,
        #                            key=k,
        #                            value=v, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


SGDTDecoderLayerType = {
    'regular': TransformerDecoderLayer,
    # 'sgdt': TransformerEncoderSGDTLayer,
    # 'sgdtv0': TransformerEncoderSGDTLayerV0NoMaskUpdate,
    # 'sgdtv1': TransformerEncoderSGDTLayerV1,
    # 'sgdt+k': TransformerEncoderSGDTLayerUpdateK,
    # 'sgdt+v': TransformerEncoderSGDTLayerUpdateV,
    #
    # 'sgdt+qkv': TransformerEncoderSGDTLayerUpdateQKV,
    # 'sgdt+mha+out': TransformerEncoderSGDTLayerUpdateMHAOut,
    # 'sgdt+mha+feature': TransformerEncoderSGDTLayerUpdateMHAFeature,
    # 'sgdt+ffn+out': TransformerEncoderSGDTLayerUpdateFFNOut,
    # 'sgdt+ffn+feature': TransformerEncoderSGDTLayerUpdateFFNFeature,

    # 'sgdtSharedAttn': TransformerEncoderSharedSelfAttnLayer,
    # 'sgdtv1FreezeSelfAttn': TransformerEncoderSGDTLayerUpdateKExtendedSelfAttn,
    # 'sgdtMarkFgBg': TransformerEncoderMarkFgBgLayer,

    # teacher student both in a single encoder layer
    'STMarkECFFNFeatureFgBgShareQV': TransformerDecoderLayerSTMarkFgBgShareQV,
    # 'parallelSTECMarkFgBgShareVNoFreeze': TransformerEncoderLayerSTMarkFgBgShareVNoFreeze,
    # 'parallelSTECShareSGDTUpdateKAttn': TransformerEncoderLayerSTShareSGDTUpdateKAttn,
    # 'parallelSTECSGDTShareVNoFreeze': TransformerEncoderLayerSTSGDTShareVNoFreeze,
}


class TransformerDecoder(nn.Module):

    def __init__(self,
                 # decoder_layer, num_layers,
                 decoder_layer=None, num_layers=None,
                 norm=None, return_intermediate=False,
                 d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                 modulate_hw_attn=False,
                 bbox_embed_diff_each_layer=False,
                 decoder_layer_list=None,  # ---
                 ):
        super().__init__()

        if decoder_layer_list is not None:
            # ------------------- TTI Modification
            self.layers = nn.ModuleList()
            self.num_layers = 0
            for l_conf in decoder_layer_list:
                decoder_layer, num_l = l_conf
                assert num_l > 0
                # nn.ModuleList
                self.layers.extend(_get_clones(decoder_layer, num_l))
                self.num_layers += num_l
            # update num_layers as it will be used later
            num_layers = self.num_layers
        else:
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


# class TransformerDecoderStandAlone(nn.Module):  ## TODO
#
#     def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
#                  d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
#                  modulate_hw_attn=False,
#                  bbox_embed_diff_each_layer=False,
#                  ):
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm
#         self.return_intermediate = return_intermediate
#         assert return_intermediate
#         self.query_dim = query_dim
#
#         assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
#         self.query_scale_type = query_scale_type
#         if query_scale_type == 'cond_elewise':
#             self.query_scale = MLP(d_model, d_model, d_model, 2)
#         elif query_scale_type == 'cond_scalar':
#             self.query_scale = MLP(d_model, d_model, 1, 2)
#         elif query_scale_type == 'fix_elewise':
#             self.query_scale = nn.Embedding(num_layers, d_model)
#         else:
#             raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
#
#         self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
#
#         self.bbox_embed = None
#         self.d_model = d_model
#         self.modulate_hw_attn = modulate_hw_attn
#         self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
#
#         if modulate_hw_attn:
#             self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
#
#         if not keep_query_pos:
#             for layer_id in range(num_layers - 1):
#                 self.layers[layer_id + 1].ca_qpos_proj = None
#
#     def forward(self, tgt, memory,
#                 tgt_mask: Optional[Tensor] = None,
#                 memory_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None,
#                 refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
#                 ):
#         output = tgt
#
#         intermediate = []
#         reference_points = refpoints_unsigmoid.sigmoid()
#         ref_points = [reference_points]
#
#         # import ipdb; ipdb.set_trace()
#
#         for layer_id, layer in enumerate(self.layers):
#             obj_center = reference_points[..., :self.query_dim]  # [num_queries, batch_size, 2]
#             # get sine embedding for the query vector
#             query_sine_embed = gen_sineembed_for_position(obj_center)
#             query_pos = self.ref_point_head(query_sine_embed)
#
#             # For the first decoder layer, we do not apply transformation over p_s
#             if self.query_scale_type != 'fix_elewise':
#                 if layer_id == 0:
#                     pos_transformation = 1
#                 else:
#                     pos_transformation = self.query_scale(output)
#             else:
#                 pos_transformation = self.query_scale.weight[layer_id]
#
#             # apply transformation
#             query_sine_embed = query_sine_embed[..., :self.d_model] * pos_transformation
#
#             # modulated HW attentions
#             if self.modulate_hw_attn:
#                 refHW_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 2
#                 query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
#                 query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)
#
#             output = layer(output, memory, tgt_mask=tgt_mask,
#                            memory_mask=memory_mask,
#                            tgt_key_padding_mask=tgt_key_padding_mask,
#                            memory_key_padding_mask=memory_key_padding_mask,
#                            pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
#                            is_first=(layer_id == 0))
#
#             # iter update
#             if self.bbox_embed is not None:
#                 if self.bbox_embed_diff_each_layer:
#                     tmp = self.bbox_embed[layer_id](output)
#                 else:
#                     tmp = self.bbox_embed(output)
#                 # import ipdb; ipdb.set_trace()
#                 tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
#                 new_reference_points = tmp[..., :self.query_dim].sigmoid()
#                 if layer_id != self.num_layers - 1:
#                     ref_points.append(new_reference_points)
#                 reference_points = new_reference_points.detach()
#
#             if self.return_intermediate:
#                 intermediate.append(self.norm(output))
#
#         if self.norm is not None:
#             output = self.norm(output)
#             if self.return_intermediate:
#                 intermediate.pop()
#                 intermediate.append(output)
#
#         if self.return_intermediate:
#             if self.bbox_embed is not None:
#                 return [
#                     torch.stack(intermediate).transpose(1, 2),
#                     torch.stack(ref_points).transpose(1, 2),
#                 ]
#             else:
#                 return [
#                     torch.stack(intermediate).transpose(1, 2),
#                     reference_points.unsqueeze(0).transpose(1, 2)
#                 ]
#
#         return output.unsqueeze(0)

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
                    src=encoder_output_list[-2]['feat'],  # memory, encoder_output_list is a list
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


class TransformerSTDecoder(TransformerDecoder):

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2

                encoder=None,
                # encoder_output_list=None,
                # second_half_encoder_layer_ids=None,
                teacher_memory=None,
                sgdt=None,
                class_embed=None,
                mask_dict=None,
                ):
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        # import ipdb; ipdb.set_trace()
        # sgdt_output_list = []
        # encoder_output_list_decoder = []
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

            if layer_id != self.num_layers - 1:  # the first N-1 layers
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                               is_first=(layer_id == 0))
            else:  # The last decoder layer
                # --------------------------------------
                # memory = teacher_memory
                # if encoder.norm is not None:
                #     memory = encoder.norm(memory)
                # output = layer(output.clone().detach(),
                #                teacher_memory, tgt_mask=tgt_mask,
                #                memory_mask=memory_mask,  # None
                #                tgt_key_padding_mask=tgt_key_padding_mask,  # None
                #                memory_key_padding_mask=memory_key_padding_mask,
                #                pos=pos,  # require_grad = False
                #                query_pos=query_pos.clone().detach(),
                #                query_sine_embed=query_sine_embed.clone().detach(),
                #                is_first=(layer_id == 0))
                output = layer(output,
                               teacher_memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,  # None
                               tgt_key_padding_mask=tgt_key_padding_mask,  # None
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos,  # require_grad = False
                               query_pos=query_pos,
                               query_sine_embed=query_sine_embed,
                               is_first=(layer_id == 0))
                # --------------------------------------

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
                    # sgdt_output_list,
                    # encoder_output_list_decoder
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2),
                    # sgdt_output_list,
                    # encoder_output_list_decoder
                ]

        return output.unsqueeze(0)


# Transformer
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
                 decoder_layer_config=None,
                 align_encoder_decoder_layers_num=None,
                 # marking_encoder_feature_by_fg1_bg0=False
                 ):

        super().__init__()

        # self.encoder_without_pos = encoder_without_pos
        # self.marking_encoder_feature_by_fg1_bg0 = marking_encoder_feature_by_fg1_bg0
        if encoder_decoder_config is None:
            encoder_type, decoder_type = TransformerEncoder, TransformerDecoder
        elif encoder_decoder_config == 'sgdt':
            encoder_type, decoder_type = TransformerSGDTEncoder, TransformerSGDTDecoder
        elif encoder_decoder_config == 'self_distillation':
            encoder_type, decoder_type = TransformerSGDTEncoder, FeatureDistillationTransformerDecoder
        elif encoder_decoder_config == 'online_distillation':
            encoder_type, decoder_type = TransformerEncoder, TransformerSTDecoder
        else:
            raise NotImplementedError

        # # self.reclaim_padded_region = reclaim_padded_region
        assert encoder_layer_config is not None and isinstance(encoder_layer_config, str)
        if len(encoder_layer_config) == 0:
            self.encoder = TransformerEmptyEncoder()
        else:
            # encoder_layer_config: 'regular_6',  'regular_4-sgdtv1_1-sgdt_1'
            encoder_layer_conf_list = parser_encoder_decoder_layers(encoder_layer_config)

            encoder_layer_list = []
            for l_type, num_l in encoder_layer_conf_list:
                assert l_type in SGDTEncoderLayerType and num_l > 0
                encoder_layer = SGDTEncoderLayerType[l_type](d_model, nhead, dim_feedforward,
                                                             dropout, activation, normalize_before,
                                                             )
                encoder_layer_list.append([encoder_layer, num_l])

            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = encoder_type(encoder_layer_list, encoder_norm)

        ###################### Configuration for decoder ##################

        decoder_norm = nn.LayerNorm(d_model)
        if decoder_layer_config is not None:
            decoder_layer_conf_list = parser_encoder_decoder_layers(decoder_layer_config)
            decoder_layer_list = []
            for l_type, num_l in decoder_layer_conf_list:
                assert l_type in SGDTDecoderLayerType and num_l > 0
                decoder_layer = SGDTDecoderLayerType[l_type](
                    d_model, nhead, dim_feedforward,
                    dropout, activation, normalize_before,
                    keep_query_pos=keep_query_pos
                )
                decoder_layer_list.append([decoder_layer, num_l])
            self.decoder = decoder_type(
                decoder_layer_list=decoder_layer_list,
                norm=decoder_norm,
                return_intermediate=return_intermediate_dec,
                d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos,
                query_scale_type=query_scale_type,
                modulate_hw_attn=modulate_hw_attn,
                bbox_embed_diff_each_layer=bbox_embed_diff_each_layer,
            )
        else:

            # Below are same with the original decoder.
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before,
                                                    keep_query_pos=keep_query_pos)
            self.decoder = decoder_type(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec,
                                        d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos,
                                        query_scale_type=query_scale_type,
                                        modulate_hw_attn=modulate_hw_attn,
                                        bbox_embed_diff_each_layer=bbox_embed_diff_each_layer,
                                        )

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
        skip_teacher_model_decoder_forward = kwargs.pop('skip_teacher_model_decoder_forward', False)

        memory, sgdt_output_list, pos_embed, mask, encoder_output_list = self.encoder(
            src, src_key_padding_mask=mask, pos=pos_embed,
            sgdt=sgdt, **kwargs,
        )

        # If this model is used as teacher, sometimes we do not need to go through decoder.
        if skip_teacher_model_decoder_forward:
            return None, None, sgdt_output_list, encoder_output_list

        # # if training token score only, we can skip the forward propagation of decoder.
        # if self.training and self.train_token_scoring_only:
        #     return None, None, sgdt_output_list

        if self.num_patterns > 0:
            l = tgt.shape[0]
            tgt[l - self.num_queries * self.num_patterns:] += \
                self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1)

        # ===================
        # manually set the fg, bg locations to be 1, 0, respectively for the last feature channel
        # to generate the mark.
        # if self.marking_encoder_feature_by_fg1_bg0:
        #     # memory: N, B, C torch.Size([756, 2, 256]);  sgdt['fg_gt']: N, B shape torch.Size([756, 2])
        #     memory[:, :, -1] = memory[:, :, -1] * 0 + sgdt.sgdt_targets['fg_gt'].type(memory.dtype)

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
                 # marking_encoder_feature_by_fg1_bg0=False,
                 decoder_layer_config=None,
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
                         encoder_decoder_config='sgdt',
                         decoder_layer_config=decoder_layer_config,
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
                 # marking_encoder_feature_by_fg1_bg0=False,
                 decoder_layer_config=None,
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
                         encoder_decoder_config='self_distillation',
                         decoder_layer_config=decoder_layer_config,
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


class TransformerOnlineSelfDistiller(Transformer):
    def __init__(self, d_model=512, nhead=8, num_queries=300, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_intermediate_dec=False, query_dim=4,
                 keep_query_pos=False, query_scale_type='cond_elewise', num_patterns=0, modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,

                 encoder_layer_config='regular_6',
                 align_encoder_decoder_layers_num=None,
                 # marking_encoder_feature_by_fg1_bg0=False,
                 decoder_layer_config=None,
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
                         encoder_decoder_config='online_distillation',
                         decoder_layer_config=decoder_layer_config,
                         )

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
            # encoder_layer_ids=self.first_half_encoder_layer_ids,
            sgdt=sgdt,
        )

        assert 'output_online_teacher' in encoder_output_list[-1]
        teacher_memory = encoder_output_list[-1]['output_online_teacher']

        if self.num_patterns > 0:
            l = tgt.shape[0]
            tgt[l - self.num_queries * self.num_patterns:] += \
                self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1)

        # hs, references = self.decoder(tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
        #                               pos=pos_embed, refpoints_unsigmoid=refpoint_embed)
        hs, references = self.decoder(
            tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask,
            pos=pos_embed, refpoints_unsigmoid=refpoint_embed,
            encoder=self.encoder,
            # second_half_encoder_layer_ids=self.second_half_encoder_layer_ids,
            sgdt=sgdt,
            mask_dict=mask_dict,
            class_embed=class_embed,
            teacher_memory=teacher_memory,
        )

        return hs, references, sgdt_output_list, encoder_output_list


def build_transformer(args):
    if args.online_self_distill_transformer:
        transformer = TransformerOnlineSelfDistiller
    elif args.feature_attn_distillation is not None:
        if args.feature_attn_distillation == 'parallel':
            transformer = TransformerSelfDistillation
        elif args.feature_attn_distillation == 'cascade':
            transformer = TransformerSGDT
        elif args.feature_attn_distillation == 'separate_trained_model':
            transformer = Transformer
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
        decoder_layer_config=args.decoder_layer_config,
        # marking_encoder_feature_by_fg1_bg0=args.marking_encoder_feature_by_fg1_bg0,
        # train_token_scoring_only=args.train_token_scoring_only
    )
