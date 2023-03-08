import warnings
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from .attn import (
    MultiheadAttention,
    MultiheadAttentionShareV,
    MultiheadAttentionShareA,
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
        ksgt_output = ksgt(x=src,)
        src_with_gt = ksgt_output['x']
    elif token_masking == 'MarkFg1Bg0':
        src_with_gt = mark_encoder_feature_by_fg_gt(src.clone(), ksgt)
    else:
        raise NotImplementedError
    return src_with_gt


def get_self_attn_q_k_v(src, pos, ksgt=None, ):
    if ksgt is None:
        # '[MHA_Out', 'FFN_Out'])
        q = k = with_pos_embed(src, pos)
        v = src
    else:
        if ksgt.encoder_token_masking and ksgt.encoder_token_masking_loc:
            token_masking = ksgt.encoder_token_masking
            token_masking_loc = ksgt.encoder_token_masking_loc

            if token_masking_loc == 'X':
                src_with_gt = get_token_gt_masking(
                    src=src, token_masking=token_masking, ksgt=ksgt)
                q = k = with_pos_embed(src_with_gt, pos)
                v = src_with_gt
            elif token_masking_loc == 'Q':
                k = with_pos_embed(src, pos)
                src_with_gt = get_token_gt_masking(src=src, token_masking=token_masking, ksgt=ksgt)
                q = with_pos_embed(src_with_gt, pos)
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


def get_cross_attn_k_v(key, value, ksgt):
    """"""
    assert torch.equal(key, value), 'Currently only support the case in which key and value are equal.'
    if ksgt is None:
        raise NotImplementedError
    else:
        if ksgt.decoder_token_masking and ksgt.decoder_token_masking_loc:
            token_masking = ksgt.decoder_token_masking
            token_masking_loc = ksgt.decoder_token_masking_loc

            if token_masking_loc == 'K':
                k = get_token_gt_masking(src=key, token_masking=token_masking, ksgt=ksgt)
                v = value
            elif token_masking_loc == 'V':
                k = key
                v = get_token_gt_masking(src=value, token_masking=token_masking, ksgt=ksgt)
            elif token_masking_loc == 'KV':
                if torch.equal(key, value):  # Usually key and value are equal.
                    k = v = get_token_gt_masking(src=key, token_masking=token_masking, ksgt=ksgt)
                else:
                    k = get_token_gt_masking(src=key, token_masking=token_masking, ksgt=ksgt)
                    v = get_token_gt_masking(src=value, token_masking=token_masking, ksgt=ksgt)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    return k, v


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

        # Sometimes, we need the attn map logits (not used
        # in KS-DETR, but for other work, e.g., attention distillation),
        # which will not be returned in the current version of nn.MultiheadAttention,
        self.attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,
            **kwargs,
        )

        self.proj_drop = nn.Dropout(proj_drop)

    # TODO: debug this function, not tested yet.
    def identity_mapping(self, output, identity):
        if torch.is_tensor(output):
            if not self.batch_first:
                output = output.permute(1, 0, 2)
            output = [identity + self.dropout(output)]
        else:
            for k, single_out in enumerate(output):
                if not self.batch_first:
                    single_out = single_out.permute(1, 0, 2)
                output[k] = identity + self.dropout(single_out)
        return output

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


class KSBaseMultiheadMultiAttention(KSBaseMultiheadAttention):
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
                           **kwargs,
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
            need_weights=True, average_attn_weights=False,
            **kwargs,
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


class KSMultiheadDualAttentionShareV(KSBaseMultiheadMultiAttention):
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

        self.attn = MultiheadAttentionShareV(  # attn_module
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,
            **kwargs,
        )


class KSMultiheadDualAttentionShareA(KSBaseMultiheadMultiAttention):
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

        self.attn = MultiheadAttentionShareA(  # attn_module
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,
            **kwargs,
        )


class KSMultiheadTripleAttentionShareAV(KSBaseMultiheadMultiAttention):
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

        # -----------
        attn_output_weight_logits = attn.clone()
        # ---------

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        output = [identity + self.proj_drop(out)]
        self_attn_out_dict = dict(sa_attn_map_logits=attn_output_weight_logits)

        return output, self_attn_out_dict


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
            # Why conduct special operation for the first layer, because the query embedding is randomly
            # initialized, and self-attn in the first layer can be removed. So the first operation
            # in the first decoder layer is actually cross-attn.
            query_pos = self.query_pos_proj(query_pos)  # query_pos is only used in the first layer.
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

        # -----------
        attn_output_weight_logits = attn.clone()
        # ---------

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        output = [identity + self.proj_drop(out)]
        cross_attn_out_dict = dict(ca_attn_map_logits=attn_output_weight_logits)

        return output, cross_attn_out_dict


class KSConditionalCrossAttentionMultiAttnBase(nn.Module):
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
        super().__init__()

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

        # -----
        self.teacher_key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.teacher_value_proj = nn.Linear(embed_dim, embed_dim)

    def get_teacher_kv(self,
                       query,
                       key=None,
                       value=None,
                       key_pos_projected=None,
                       is_first_layer=False,
                       ):
        key_content = self.teacher_key_content_proj(key)
        value = self.teacher_value_proj(value)

        # ----
        N, B, C = query.shape  # query_content.shape
        # ----
        HW, _, _ = key_content.shape

        # position projection
        # key_pos = self.key_pos_proj(key_pos)
        key_pos = key_pos_projected
        if is_first_layer:
            k = key_content + key_pos
        else:
            k = key_content
        v = value

        # preprocess
        k = k.view(HW, B, self.num_heads, C // self.num_heads)  # N, 16, 256
        key_pos = key_pos.view(HW, B, self.num_heads, C // self.num_heads)
        k = torch.cat([k, key_pos], dim=3).view(HW, B, C * 2)

        # attention calculation
        k = k.reshape(HW, B, self.num_heads, C * 2 // self.num_heads).permute(1, 2, 0, 3)
        v = v.reshape(HW, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        return k, v

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
        pass


class KSConditionalCrossAttentionDynamicAttn(KSConditionalCrossAttentionMultiAttnBase):
    """Used as KSConditionalCrossAttentionTripleAttn in default, but can be set to other attention in
    each iteration.

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

        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            batch_first=batch_first,
            **kwargs,
        )

        # super().__init__()
        #
        # self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        # self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        # self.query_pos_sine_proj = nn.Linear(embed_dim, embed_dim)
        # self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        # self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        # self.value_proj = nn.Linear(embed_dim, embed_dim)
        # self.out_proj = nn.Linear(embed_dim, embed_dim)
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj_drop = nn.Dropout(proj_drop)
        # self.num_heads = num_heads
        # self.batch_first = batch_first
        #
        # # -----
        # self.teacher_key_content_proj = nn.Linear(embed_dim, embed_dim)
        # self.teacher_value_proj = nn.Linear(embed_dim, embed_dim)

        # DYNAMIC_ATTN_LIST = ['TripleAttention', 'PlainAttention', 'DualAttentionShareV', 'DualAttentionShareA', ]
        # We can use self.dynamic_attn_config to set fixed attention. For instance,
        # if self.dynamic_attn_config = TripleAttention, then this dynamic attention is used as TripleAttention.
        self.default_dynamic_attn_config = 'TripleAttention'

    def set_dynamic_attn_config(self, **kwargs):
        # if the dynamic_attn_config is not explicitly set, then we the default setting self.default_dynamic_attn_config
        dynamic_attn_config = kwargs.get('dynamic_attn_config', None)
        if dynamic_attn_config is None:
            dynamic_attn_config = self.default_dynamic_attn_config

        assert dynamic_attn_config is not None
        return dynamic_attn_config

    # def set_dynamic_attn_config(self, **kwargs):
    #     # For dynamic attention, self.default_dynamic_attn_config will have no effect on it.
    #     dynamic_attn_config = kwargs.get('dynamic_attn_config', None)
    #
    #     # assert dynamic_attn_config is not None
    #     # If dynamic_attn_config is not set, we use PlainAttention in default.
    #     if dynamic_attn_config is None:
    #         dynamic_attn_config = 'PlainAttention'
    #
    #     return dynamic_attn_config

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
        # Back up the key and value for later use.
        key_copy, value_copy = key.clone(), value.clone()
        # -----------------
        # student branch
        # -----------------
        # content projection
        query_content = self.query_content_proj(query)
        key_content = self.key_content_proj(key)
        value = self.value_proj(value)

        # shape info
        N, B, C = query_content.shape
        HW, _, _ = key_content.shape

        # position projection
        key_pos = self.key_pos_proj(key_pos)
        # ----------
        key_pos_projected = key_pos.clone()
        # ----------
        if is_first_layer:
            # Why conduct special operation for the first layer, because the query embedding is randomly
            # initialized, and self-attn in the first layer can be removed. So the first operation
            # in the first decoder layer is actually cross-attn.
            query_pos = self.query_pos_proj(query_pos)  # query_pos is only used in the first layer.
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
        # -----------
        attn_output_weight_logits = attn.clone()
        # ---------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out_student = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out_student = self.out_proj(out_student)

        # ----------------------------
        # Process teacher branches
        # ----------------------------
        ksgt = kwargs.get('ksgt', None)  # key_copy, value_copy (value has been modified above)
        dynamic_attn_config = self.set_dynamic_attn_config(**kwargs)

        # if dynamic_attn_config == 'PlainAttention':

        if ksgt is None or dynamic_attn_config == 'PlainAttention':
            # The teacher branch of subset_layers_forward() will be ignored and all the multi-attention
            # will be used as plain attention.
            key_teacher, value_teacher = key_copy, value_copy
        else:
            assert ksgt is not None
            key_teacher, value_teacher = get_cross_attn_k_v(key=key_copy, value=value_copy, ksgt=ksgt)
        # The multi-attn branch (no matter set to 'PlainAttention' or not), will conduct the following line:
        k_teacher, v_teacher = self.get_teacher_kv(
            query=query,
            key=key_teacher,
            value=value_teacher,
            key_pos_projected=key_pos_projected,
            is_first_layer=is_first_layer,
        )

        attn_output_weights_logits_t, out = [], []
        if dynamic_attn_config == 'PlainAttention':  # q_teacher, k_teacher, v_teacher not used.
            if k_teacher is not None or v_teacher is not None:
                not_used = 0
                if k_teacher is not None:
                    not_used = 0 * k_teacher.sum()
                if v_teacher is not None:
                    not_used = not_used + 0 * v_teacher.sum()
                out_student = out_student + not_used

            out = [out_student]

        elif dynamic_attn_config in ['DualAttentionShareV', 'TripleAttention']:
            # ----------- teacher 1 (share V)
            attn_teacher = q @ k_teacher.transpose(-2, -1)
            attn_output_weights_logits_teacher = attn_teacher.clone()
            attn_teacher = attn_teacher.softmax(dim=-1)  # torch.Size([2, 8, 300, 540])
            attn_teacher = self.attn_drop(attn_teacher)
            out_teacher_share_v = (attn_teacher @ v).transpose(1, 2).reshape(B, N, C)
            out_teacher_share_v = self.out_proj(out_teacher_share_v)

            if dynamic_attn_config == 'DualAttentionShareV':
                # TODO: assert self.
                if v_teacher is not None:
                    not_used = 0 * v_teacher.sum()  # v_teacher not used.
                    out_teacher_share_v = out_teacher_share_v + not_used

                attn_output_weights_logits_t += [attn_output_weights_logits_teacher]
                out = [out_student, out_teacher_share_v]
            if dynamic_attn_config == 'TripleAttention':
                # --------- teacher 2 (share A)
                out_teacher_share_A = (attn @ v_teacher).transpose(1, 2).reshape(B, N, C)
                out_teacher_share_A = self.out_proj(out_teacher_share_A)
                # ---------
                out = [out_student, out_teacher_share_v, out_teacher_share_A]
                attn_output_weights_logits_t += [attn_output_weights_logits_teacher, attn_output_weight_logits]

        elif dynamic_attn_config == 'DualAttentionShareA':
            out_teacher_share_A = (attn @ v_teacher).transpose(1, 2).reshape(B, N, C)
            out_teacher_share_A = self.out_proj(out_teacher_share_A)

            if k_teacher is not None:
                not_used = 0 * k_teacher.sum()  # q_teacher, k_teacher not used.
                out_teacher_share_A = out_teacher_share_A + not_used

            out = [out_student, out_teacher_share_A]
            attn_output_weights_logits_t += [attn_output_weight_logits]

        else:
            raise NotImplementedError

        assert len(attn_output_weights_logits_t) == len(out) - 1

        if not self.batch_first:
            # out = out.transpose(0, 1)
            out = [single_out.transpose(0, 1) for single_out in out]

        output = [identity + self.proj_drop(single_out) for single_out in out]

        cross_attn_out_dict = dict(ca_attn_map_logits=attn_output_weight_logits,)

        if len(attn_output_weights_logits_t) > 0:
            cross_attn_out_dict.update(dict(attn_output_weights_logits_t=attn_output_weights_logits_t))

        return output, cross_attn_out_dict


class KSConditionalCrossAttentionTripleAttn(KSConditionalCrossAttentionDynamicAttn):

    def __init__(
            self,
            embed_dim,
            num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
            batch_first=False,
            **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            batch_first=batch_first,
            **kwargs,
        )

        # # -----
        self.teacher_key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.teacher_value_proj = nn.Linear(embed_dim, embed_dim)
        self.default_dynamic_attn_config = 'TripleAttention'
        
        
class KSConditionalCrossAttentionShareA(KSConditionalCrossAttentionDynamicAttn):

    def __init__(
            self,
            embed_dim,
            num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
            batch_first=False,
            **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            batch_first=batch_first,
            **kwargs,
        )

        # -----
        self.teacher_key_content_proj = None
        self.teacher_value_proj = nn.Linear(embed_dim, embed_dim)
        self.default_dynamic_attn_config = 'DualAttentionShareA'

    def get_teacher_kv(self,
                       query,
                       key=None,
                       value=None,
                       key_pos_projected=None,
                       is_first_layer=False,
                       ):
        # key_content = self.teacher_key_content_proj(key)
        value = self.teacher_value_proj(value)

        # shape info
        # ----
        N, B, C = query.shape  # query_content.shape
        HW, _, _ = key.shape
        # ----
        # HW, _, _ = key_content.shape

        # # position projection
        # # key_pos = self.key_pos_proj(key_pos)
        # key_pos = key_pos_projected
        # if is_first_layer:
        #     k = key_content + key_pos
        # else:
        #     k = key_content
        v = value

        # # preprocess
        # k = k.view(HW, B, self.num_heads, C // self.num_heads)  # N, 16, 256
        # key_pos = key_pos.view(HW, B, self.num_heads, C // self.num_heads)
        # k = torch.cat([k, key_pos], dim=3).view(HW, B, C * 2)

        # attention calculation
        # k = k.reshape(HW, B, self.num_heads, C * 2 // self.num_heads).permute(1, 2, 0, 3)
        v = v.reshape(HW, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)

        k = None
        return k, v


class KSConditionalCrossAttentionShareV(KSConditionalCrossAttentionDynamicAttn):

    def __init__(
            self,
            embed_dim,
            num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
            batch_first=False,
            **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            batch_first=batch_first,
            **kwargs,
        )

        # -----
        self.teacher_key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.teacher_value_proj = None
        self.default_dynamic_attn_config = 'DualAttentionShareV'

    def get_teacher_kv(self,
                       query,
                       key=None,
                       value=None,
                       key_pos_projected=None,
                       is_first_layer=False,
                       ):
        key_content = self.teacher_key_content_proj(key)
        # value = self.teacher_value_proj(value)

        # shape info
        # ----
        N, B, C = query.shape  # query_content.shape
        # ----
        HW, _, _ = key_content.shape

        # position projection
        # key_pos = self.key_pos_proj(key_pos)
        key_pos = key_pos_projected
        if is_first_layer:
            k = key_content + key_pos
        else:
            k = key_content
        # v = value

        # preprocess
        k = k.view(HW, B, self.num_heads, C // self.num_heads)  # N, 16, 256
        key_pos = key_pos.view(HW, B, self.num_heads, C // self.num_heads)
        k = torch.cat([k, key_pos], dim=3).view(HW, B, C * 2)

        # # attention calculation
        k = k.reshape(HW, B, self.num_heads, C * 2 // self.num_heads).permute(1, 2, 0, 3)
        # v = v.reshape(HW, B, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
        return k, None




