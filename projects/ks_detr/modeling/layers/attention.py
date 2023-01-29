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
# ------------------------------------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py
# ------------------------------------------------------------------------------------------------

import warnings
from typing import Optional, List
import torch
import torch.nn as nn
from torch import Tensor

from .attn import (
    MultiheadAttentionSeparateWeight,
    MultiheadAttention,
    MultiheadDualAttention,
    MultiheadAttentionShareVOutProj,
    MultiheadAttentionShareAttnOutProj,
    MultiheadTripleAttention,
)


def mark_encoder_feature_by_fg_gt(memory, ksgt):
    # set the last feature dimension to be the ft_gt mask
    assert isinstance(ksgt.ksgt_targets, dict) and 'fg_gt' in ksgt.ksgt_targets
    # memory: N, B, C torch.Size([756, 2, 256]);  ksgt['fg_gt']: N, B shape torch.Size([756, 2])
    memory[:, :, -1] = memory[:, :, -1] * 0 + ksgt.ksgt_targets['fg_gt'].type(memory.dtype)
    return memory


def with_pos_embed(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos


def get_token_gt_masking(src, token_masking, ksgt):
    if token_masking == 'sMLP':
        ksgt_output = ksgt(x=src,
                           # mask=ksgt.token_mask,
                           # ksgt_targets=ksgt_targets,
                           # feat_map_size=feat_map_size,  # (h, w)
                           # sigma=sigma,
                           )
        src_with_gt = ksgt_output['x']  # with_pos_embed(ksgt_output['x'], pos=k_adapted_pos)
    elif token_masking == 'MarkFg1Bg0':
        src_with_gt = mark_encoder_feature_by_fg_gt(src.clone(), ksgt)
    else:
        raise NotImplementedError
    return src_with_gt


def get_self_attn_q_k_v(src, pos, ksgt=None, ):
    # q = k = v = None  # This is a big bug, never use it in this way.

    # q, k, v = None, None, None
    # assert token_masking and token_masking_loc in ['X', 'Q', 'K', 'V', 'QK', 'KV',]
    # This should not be calculated in advance. and Never use None to src_with_gt, which will cause loss
    # inconsistent.
    # src_with_gt = None
    if ksgt is None:
        # '[MHA_Out', 'FFN_Out'])
        q = k = with_pos_embed(src, pos)
        v = src
    else:
        if ksgt.token_masking and ksgt.token_masking_loc:
            token_masking = ksgt.token_masking
            token_masking_loc = ksgt.token_masking_loc
            # src_key_padding_mask = ksgt.src_key_padding_mask
            if token_masking_loc == 'X':
                src_with_gt = get_token_gt_masking(
                    src=src, token_masking=token_masking, ksgt=ksgt)
                q = k = with_pos_embed(src_with_gt, pos)
                v = src_with_gt
            elif token_masking_loc == 'Q':
                k = with_pos_embed(src, pos)
                src_with_gt = get_token_gt_masking(src=src, token_masking=token_masking, ksgt=ksgt)
                q = with_pos_embed(src_with_gt, pos)

                # here it is safe to use v = src, as the src here will not affect the src passed in
                # (outside of this function)
                v = src
            elif token_masking_loc == 'K':
                q = with_pos_embed(src, pos)
                src_with_gt = get_token_gt_masking(src=src, token_masking=token_masking, ksgt=ksgt)
                k = with_pos_embed(src_with_gt, pos)
                v = src
            elif token_masking_loc == 'V':
                q = k = with_pos_embed(src, pos)
                src_with_gt = get_token_gt_masking(src=src, token_masking=token_masking, ksgt=ksgt)
                v = src_with_gt
            elif token_masking_loc == 'QK':
                src_with_gt = get_token_gt_masking(src=src, token_masking=token_masking, ksgt=ksgt)
                q = k = with_pos_embed(src_with_gt, pos)
                v = src
            elif token_masking_loc == 'KV':
                q = with_pos_embed(src, pos)
                src_with_gt = get_token_gt_masking(src=src, token_masking=token_masking, ksgt=ksgt)
                k = with_pos_embed(src_with_gt, pos)
                v = src_with_gt
            else:  # token_masking_loc  ['MHA_out', 'MHA_feature', 'FFN_out', 'FFN_feature',]:
                raise NotImplementedError
        else:
            q = k = with_pos_embed(src, pos)
            v = src

    return q, k, v


class KSBaseMultiheadAttention(nn.Module):
    """A wrapper for my implemented MultiheadAttention that returnes attn_logits.

    ``torch.nn.MultiheadAttention``
    Implemente MultiheadAttention with identity connection,
    and position embedding is also passed as input.

    Args:
        embed_dim (int): The embedding dimension for attention.
        num_heads (int): The number of attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `MultiheadAttention`.
            Default: 0.0.
        batch_first (bool): if `True`, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            batch_first: bool = False,
            **kwargs,
    ):
        super(KSBaseMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        # self.attn = nn.MultiheadAttention(
        #     embed_dim=embed_dim,
        #     num_heads=num_heads,
        #     dropout=attn_drop,
        #     batch_first=batch_first,
        #     **kwargs,
        # )

        # We need the attn map logits, which will not be returned in the current version of nn.MultiheadAttention,
        # also we want the control the w_q, w_k, w_v separately for distillation, or attention, Value sharing, which
        # is difficult in the nn.MultiheadAttention, which use packed w, b.
        # TODO: change to MultiheadAttentionSeparateWeight
        # attn_module = kwargs.pop('self_attn_module', MultiheadAttention)

        self.attn = MultiheadAttention(  # attn_module
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,
            **kwargs,
        )
        # the following parameters are not specified,
        #           bias = True, add_bias_kv = False, add_zero_attn = False,
        #           kdim = None, vdim = None,  device = None, dtype = None
        # but there is no problem, because self.attn is usually initialized by
        #           self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            identity: Optional[torch.Tensor] = None,
            query_pos: Optional[torch.Tensor] = None,
            key_pos: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        """Forward function for `MultiheadAttention`

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as x, will
                be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is" f"missing in {self.__class__.__name__}."
                    )
        if query_pos is not None:
            query = query + query_pos  # This does not modify identity, even identity = query before.
        if key_pos is not None:
            key = key + key_pos

            # out = self.attn(
            #     query=query,
            #     key=key,
            #     value=value,
            #     attn_mask=attn_mask,
            #     key_padding_mask=key_padding_mask,
            #
            # )[0]

        out, _, attn_output_weight_logits = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,

            # ----
            need_weights=True, average_attn_weights=False
        )
        self_attn_out_dict = dict(attn_map_logits=attn_output_weight_logits)

        if torch.is_tensor(out):
            output = [identity + self.proj_drop(out)]
        else:
            output = [identity + self.proj_drop(single_out) for single_out in out]

        return output, self_attn_out_dict


class KSBaseMultiheadAttentionSeparateWeight(KSBaseMultiheadAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            batch_first: bool = False,
            **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop,
            proj_drop=proj_drop, batch_first=batch_first, **kwargs,
        )

        self.attn = MultiheadAttentionSeparateWeight(  # attn_module
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,
            **kwargs,
        )


class KSMultiheadAttentionWithGT(KSBaseMultiheadAttention):
    """Multi head Attention with gt mask input.
    """

    def forward(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            identity: Optional[torch.Tensor] = None,
            query_pos: Optional[torch.Tensor] = None,
            key_pos: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            **kwargs,

    ):  # -> torch.Tensor
        """ key, key_pos, value are not used even they are fed in.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as x, will
                be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """

        if identity is None:
            identity = query

        ksgt = kwargs.get('ksgt', None)
        q, k, v = get_self_attn_q_k_v(src=query, pos=query_pos, ksgt=ksgt)

        out, _, attn_output_weight_logits = self.attn(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            # ----
            need_weights=True, average_attn_weights=False
        )
        self_attn_out_dict = dict(attn_map_logits=attn_output_weight_logits)

        if torch.is_tensor(out):
            output = [identity + self.proj_drop(out)]
        else:
            output = [identity + self.proj_drop(single_out) for single_out in out]

        return output, self_attn_out_dict


class KSBaseMultiheadDualAttention(KSBaseMultiheadAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            batch_first: bool = False,
            **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop,
            proj_drop=proj_drop, batch_first=batch_first, **kwargs,
        )
        del self.attn  # The self.attn must be initialized by child class

    def forward_multi_attn(self,
                           query: torch.Tensor,
                           key: torch.Tensor,
                           value: torch.Tensor,
                           identity: torch.Tensor,
                           query_teacher: Optional[torch.Tensor] = None,
                           key_teacher: Optional[torch.Tensor] = None,
                           value_teacher: Optional[torch.Tensor] = None,

                           attn_mask: Optional[torch.Tensor] = None,
                           key_padding_mask: Optional[torch.Tensor] = None,
                           ):
        # query_teacher, key_teacher, value_teacher are all fed into self.attn, but based on the dual_attn type, they
        # might not be used. For instance, share_v dual attention will not use value_teacher, share_attn dual attn will
        # not use query_teacher and key_teacher.
        out, _, attn_output_weight_logits = self.attn(
            query=query,
            key=key,
            value=value,
            query_teacher=query_teacher,
            key_teacher=key_teacher,
            value_teacher=value_teacher,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            # ----
            need_weights=True, average_attn_weights=False
        )
        self_attn_out_dict = dict(attn_map_logits=attn_output_weight_logits)

        if torch.is_tensor(out):
            output = [identity + self.proj_drop(out)]
        else:
            output = [identity + self.proj_drop(single_out) for single_out in out]

        return output, self_attn_out_dict

    def forward(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            identity: Optional[torch.Tensor] = None,
            query_pos: Optional[torch.Tensor] = None,
            key_pos: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            **kwargs,

    ):
        """ key, key_pos, value are not used even they are fed in.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as x, will
                be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """

        if identity is None:
            identity = query

        ksgt = kwargs.get('ksgt', None)
        q, k, v = get_self_attn_q_k_v(src=query, pos=query_pos, ksgt=None)
        q_teacher, k_teacher, v_teacher = get_self_attn_q_k_v(src=query, pos=query_pos, ksgt=ksgt)

        return self.forward_multi_attn(
            query=q,
            key=k,
            value=v,
            query_teacher=q_teacher,
            key_teacher=k_teacher,
            value_teacher=v_teacher,
            identity=identity,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )


class KSMultiheadDualAttentionShareVOutProjV0(KSBaseMultiheadDualAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            batch_first: bool = False,
            **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop,
            proj_drop=proj_drop, batch_first=batch_first, **kwargs,
        )

        self.attn = MultiheadAttentionShareVOutProj(  # attn_module
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,
            **kwargs,
        )


class KSMultiheadDualAttentionShareAttnOutProjV0(KSBaseMultiheadDualAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            batch_first: bool = False,
            **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop,
            proj_drop=proj_drop, batch_first=batch_first, **kwargs,
        )
        # del self.attn
        self.attn = MultiheadAttentionShareAttnOutProj(  # attn_module
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,
            **kwargs,
        )


class KSMultiheadTripleAttentionQKVShareAttnOutProjV0(KSBaseMultiheadDualAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            batch_first: bool = False,
            **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop,
            proj_drop=proj_drop, batch_first=batch_first, **kwargs,
        )
        # del self.attn
        self.attn = MultiheadTripleAttention(  # attn_module
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,
            **kwargs,
        )

    def forward(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            identity: Optional[torch.Tensor] = None,
            query_pos: Optional[torch.Tensor] = None,
            key_pos: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        """ key, key_pos, value are not used even they are fed in.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as x, will
                be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """

        if identity is None:
            identity = query

        # # =======================  I used this in my experiments for ICML, but this does not support
        # Marking experiments for ablation study.

        # ksgt = kwargs.get('ksgt', None)
        # assert ksgt is not None
        # # torch.Size([850, 2, 256]) (N, B, C)
        # q = k = with_pos_embed(query, pos=query_pos)
        # ksgt_output = ksgt(x=query,)
        # src_with_gt = ksgt_output['x']
        # q_teacher = k_teacher = with_pos_embed(src_with_gt, pos=query_pos)
        # v_teacher = src_with_gt
        # # =======================

        # q, k, v = get_self_attn_q_k_v(src=query, pos=query_pos, ksgt=None)
        # q_teacher, k_teacher, v_teacher = get_self_attn_q_k_v(src=query, pos=query_pos, ksgt=ksgt)

        ksgt = kwargs.get('ksgt', None)
        q, k, _ = get_self_attn_q_k_v(src=query, pos=query_pos, ksgt=None)
        q_teacher, k_teacher, v_teacher = get_self_attn_q_k_v(src=query, pos=query_pos, ksgt=ksgt)

        return self.forward_multi_attn(
            query=q,
            key=k,
            value=query,  # v
            query_teacher=q_teacher,
            key_teacher=k_teacher,
            value_teacher=v_teacher,
            identity=identity,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )


class KSMultiheadDualAttentionShareVOutProj(KSBaseMultiheadDualAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            batch_first: bool = False,
            **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop,
            proj_drop=proj_drop, batch_first=batch_first, **kwargs,
        )

        self.attn = MultiheadDualAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,

            share_v=True, share_out_proj_weight=True,  # This is for ShareVOutProj
            **kwargs,
        )
        # self.self_attn = MultiheadDualAttention(
        #     d_model, nhead, dropout=dropout,
        #     share_v=True, share_out_proj_weight=True,
        # )


class KSMultiheadDualAttentionShareAttnOutProj(KSBaseMultiheadDualAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            batch_first: bool = False,
            **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop,
            proj_drop=proj_drop, batch_first=batch_first, **kwargs,
        )

        self.attn = MultiheadDualAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,

            share_v=False, share_attn_map=True,
            share_out_proj_weight=True,
            **kwargs,
        )
        # self.self_attn = MultiheadDualAttention(
        #     d_model, nhead, dropout=dropout,
        #     share_v=False, share_attn_map=True,
        #     share_out_proj_weight=True,
        # )


# class KSMultiScaleTripleAttentionShareOutProj(KSBaseMultiheadDualAttention):
#     def __init__(
#             self,
#             embed_dim: int,
#             num_heads: int,
#             attn_drop: float = 0.0,
#             proj_drop: float = 0.0,
#             batch_first: bool = False,
#             **kwargs,
#     ):
#         super().__init__(
#             embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop,
#             proj_drop=proj_drop, batch_first=batch_first, **kwargs,
#         )
#
#         self.attn = MultiheadTripleAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             dropout=attn_drop,
#             batch_first=batch_first,
#
#             share_v=False, share_attn_map=True,
#             share_out_proj_weight=True,
#             **kwargs,
#         )
#         # self.self_attn = MultiheadDualAttention(
#         #     d_model, nhead, dropout=dropout,
#         #     share_v=False, share_attn_map=True,
#         #     share_out_proj_weight=True,
#         # )


class KSConditionalSelfAttention(nn.Module):
    """Conditional Self-Attention Module used in Conditional-DETR

    `Conditional DETR for Fast Training Convergence.
    <https://arxiv.org/pdf/2108.06152.pdf>`_


    Args:
        embed_dim (int): The embedding dimension for attention.
        num_heads (int): The number of attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `MultiheadAttention`.
            Default: 0.0.
        batch_first (bool): if `True`, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
            batch_first=False,
            **kwargs,
    ):
        super(KSConditionalSelfAttention, self).__init__()
        self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.batch_first = batch_first

    def forward(
            self,
            query,
            key=None,
            value=None,
            identity=None,
            query_pos=None,
            key_pos=None,
            attn_mask=None,
            key_padding_mask=None,
            **kwargs,
    ):
        """Forward function for `ConditionalSelfAttention`

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as `query``,
                which will be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query  # identity is None in default
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is" f"missing in {self.__class__.__name__}."
                    )

        assert (
                query_pos is not None and key_pos is not None
        ), "query_pos and key_pos must be passed into ConditionalAttention Module"

        # transpose (b n c) to (n b c) for attention calculation
        if self.batch_first:
            query = query.transpose(0, 1)  # (n b c)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            query_pos = query_pos.transpose(0, 1)
            key_pos = key_pos.transpose(0, 1)
            identity = identity.transpose(0, 1)

        # query/key/value content and position embedding projection
        query_content = self.query_content_proj(query)
        query_pos = self.query_pos_proj(query_pos)
        key_content = self.key_content_proj(key)
        key_pos = self.key_pos_proj(key_pos)
        value = self.value_proj(value)

        # attention calculation
        N, B, C = query_content.shape
        q = query_content + query_pos
        k = key_content + key_pos
        v = value

        q = q.reshape(N, B, self.num_heads, C // self.num_heads).permute(
            1, 2, 0, 3
        )  # (B, num_heads, N, head_dim)
        k = k.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        v = v.reshape(N, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # add attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float("-inf"))
            else:
                attn += attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        # if torch.is_tensor(out):
        #     output = [identity + self.proj_drop(out)]
        # else:
        #     output = [identity + self.proj_drop(single_out) for single_out in out]

        return [identity + self.proj_drop(out)]


class KSConditionalCrossAttention(nn.Module):
    """Conditional Cross-Attention Module used in Conditional-DETR

    `Conditional DETR for Fast Training Convergence.
    <https://arxiv.org/pdf/2108.06152.pdf>`_


    Args:
        embed_dim (int): The embedding dimension for attention.
        num_heads (int): The number of attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `MultiheadAttention`.
            Default: 0.0.
        batch_first (bool): if `True`, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
            batch_first=False,
            **kwargs,
    ):
        super(KSConditionalCrossAttention, self).__init__()
        self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_sine_proj = nn.Linear(embed_dim, embed_dim)
        self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_heads = num_heads
        self.batch_first = batch_first

    def forward(
            self,
            query,
            key=None,
            value=None,
            identity=None,
            query_pos=None,
            key_pos=None,
            query_sine_embed=None,
            is_first_layer=False,
            attn_mask=None,
            key_padding_mask=None,
            **kwargs,
    ):
        """Forward function for `ConditionalCrossAttention`

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as x, will
                be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            query_sine_embed (torch.Tensor): None
            is_first_layer (bool): None
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is" f"missing in {self.__class__.__name__}."
                    )

        assert (
                query_pos is not None and key_pos is not None
        ), "query_pos and key_pos must be passed into ConditionalAttention Module"

        # transpose (b n c) to (n b c) for attention calculation
        if self.batch_first:
            query = query.transpose(0, 1)  # (n b c)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            query_pos = query_pos.transpose(0, 1)
            key_pos = key_pos.transpose(0, 1)
            identity = identity.transpose(0, 1)

        # content projection
        query_content = self.query_content_proj(query)
        key_content = self.key_content_proj(key)
        value = self.value_proj(value)

        # shape info
        N, B, C = query_content.shape
        HW, _, _ = key_content.shape

        # position projection
        key_pos = self.key_pos_proj(key_pos)
        if is_first_layer:
            query_pos = self.query_pos_proj(query_pos)
            q = query_content + query_pos
            k = key_content + key_pos
        else:
            q = query_content
            k = key_content
        v = value

        # preprocess
        q = q.view(N, B, self.num_heads, C // self.num_heads)
        query_sine_embed = self.query_pos_sine_proj(query_sine_embed).view(
            N, B, self.num_heads, C // self.num_heads
        )
        q = torch.cat([q, query_sine_embed], dim=3).view(N, B, C * 2)

        k = k.view(HW, B, self.num_heads, C // self.num_heads)  # N, 16, 256
        key_pos = key_pos.view(HW, B, self.num_heads, C // self.num_heads)
        k = torch.cat([k, key_pos], dim=3).view(HW, B, C * 2)

        # attention calculation
        q = q.reshape(N, B, self.num_heads, C * 2 // self.num_heads).permute(
            1, 2, 0, 3
        )  # (B, num_heads, N, head_dim)
        k = k.reshape(HW, B, self.num_heads, C * 2 // self.num_heads).permute(1, 2, 0, 3)
        v = v.reshape(HW, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)

        scale = (C * 2 // self.num_heads) ** -0.5
        q = q * scale
        attn = q @ k.transpose(-2, -1)

        # add attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float("-inf"))
            else:
                attn += attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        # if torch.is_tensor(out):
        #     output = [identity + self.proj_drop(out)]
        # else:
        #     output = [identity + self.proj_drop(single_out) for single_out in out]

        return [identity + self.proj_drop(out)]
