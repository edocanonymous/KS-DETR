# https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import functional as F

# ------------------------
from torch import nn
import torch
import math
from torch._jit_internal import boolean_dispatch, List, Optional, _overload, Tuple

try:
    from torch.overrides import has_torch_function, handle_torch_function
except:
    from torch._overrides import has_torch_function, handle_torch_function
Tensor = torch.Tensor

from torch.nn.functional import linear, pad, softmax, dropout

# temporarly comment out .is_nested
# from .attn import _mha_shape_check
# from .self_attn import _linear


# ============================
# (w_q, w_k, w_v) and (b_q, b_k, b_v) are packed, we cannot set requires_grad = False for only
# the weights (w_q, b_q) of q, or that of k, v. So we used some tricks to get around this issue.
# Still use them in the forward graph, but make their grad = 0, so they will never be updated.
# ============================
def _linear(q, w_q, b_q):
    if q is not None and w_q is not None:  # b_q is optional
        return linear(q, w_q, b_q)
    else:
        return None


def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor,
                     key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], num_heads: int):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, \
            ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, \
                ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
    elif query.dim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, \
            ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, \
                ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")

        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert attn_mask.shape == expected_shape, \
                    (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor")

    return is_batched


class MultiheadDualAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.
    Multi-Head Attention is defined as:
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.
    ``forward()`` will use a special optimized implementation if all of the following
    conditions are met:
    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor. This
      restriction will be loosened in the future.)
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - dropout is 0
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``batch_first`` is ``True`` and the input is batched
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - at most one of ``key_padding_mask`` or ``attn_mask`` is passed
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed
    If the optimized implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.
    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
    Examples::
        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None,
                 share_q=False, share_k=False, share_v=False, share_attn_map=False,
                 share_out_proj_weight=False,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        # self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
        self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))

        self.q_in_proj_bias = self.k_in_proj_bias = self.v_in_proj_bias = None
        if bias:
            self.q_in_proj_bias = Parameter(torch.empty(embed_dim, **factory_kwargs))
            self.k_in_proj_bias = Parameter(torch.empty(embed_dim, **factory_kwargs))
            self.v_in_proj_bias = Parameter(torch.empty(embed_dim, **factory_kwargs))

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        # ------------------------------------
        self.share_q = share_q
        self.share_k = share_k
        self.share_v = share_v
        self.share_attn_map = share_attn_map
        self.share_out_proj_weight = share_out_proj_weight

        self.teacher_q_proj_weight = self.teacher_k_proj_weight = self.teacher_v_proj_weight = None
        self.teacher_q_in_proj_bias = self.teacher_k_in_proj_bias = self.teacher_v_in_proj_bias = None
        if not (self.share_q or self.share_attn_map):
            self.teacher_q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            if bias:
                self.teacher_q_in_proj_bias = Parameter(torch.empty(embed_dim, **factory_kwargs))

        if not (self.share_k or self.share_attn_map):
            self.teacher_k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            if bias:
                self.teacher_k_in_proj_bias = Parameter(torch.empty(embed_dim, **factory_kwargs))

        if not self.share_v:
            self.teacher_v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            if bias:
                self.teacher_v_in_proj_bias = Parameter(torch.empty(embed_dim, **factory_kwargs))

        if add_bias_kv and not (self.share_k or self.share_attn_map) and not self.share_v:
            self.teacher_k_bias = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.teacher_v_bias = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.teacher_k_bias = self.teacher_v_bias = None

        if self.share_out_proj_weight:
            self.teacher_out_proj = None
        else:
            self.teacher_out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        # -----------------

        self._reset_parameters()

    def _reset_parameters(self):

        def _init_proj_weight(proj_weight):
            if proj_weight is not None:
                xavier_uniform_(proj_weight)

        _init_proj_weight(self.q_proj_weight)
        _init_proj_weight(self.k_proj_weight)
        _init_proj_weight(self.v_proj_weight)
        _init_proj_weight(self.teacher_q_proj_weight)
        _init_proj_weight(self.teacher_k_proj_weight)
        _init_proj_weight(self.teacher_v_proj_weight)

        def _init_proj_bias(proj_bias):
            if proj_bias is not None:
                constant_(proj_bias, 0.)

        if self.q_in_proj_bias is not None or self.k_in_proj_bias is not None or self.v_in_proj_bias is not None:
            constant_(self.out_proj.bias, 0.)
            _init_proj_bias(self.q_in_proj_bias)
            _init_proj_bias(self.k_in_proj_bias)
            _init_proj_bias(self.v_in_proj_bias)

        if self.teacher_q_in_proj_bias is not None or self.teacher_k_in_proj_bias is not None or \
                self.teacher_v_in_proj_bias is not None:
            if self.teacher_out_proj is not None:
                constant_(self.teacher_out_proj.bias, 0.)

            _init_proj_bias(self.teacher_q_in_proj_bias)
            _init_proj_bias(self.teacher_k_in_proj_bias)
            _init_proj_bias(self.teacher_v_in_proj_bias)

        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

        if self.teacher_k_bias is not None:
            xavier_normal_(self.teacher_k_bias)
        if self.teacher_v_bias is not None:
            xavier_normal_(self.teacher_v_bias)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                query_teacher: Optional[Tensor] = None,
                key_teacher: Optional[Tensor] = None,
                value_teacher: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True,
                ):  # -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)
    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.
        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        # elif query.is_nested and key_padding_mask is not None:
        #     why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"
        elif self.num_heads % 2 == 1:
            why_not_fast_path = "num_heads is odd"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            assert why_not_fast_path
            # if not why_not_fast_path:
            #     return torch._native_multi_head_attention(
            #         query,
            #         key,
            #         value,
            #         self.embed_dim,
            #         self.num_heads,
            #         self.in_proj_weight,
            #         self.in_proj_bias,
            #         self.out_proj.weight,
            #         self.out_proj.bias,
            #         key_padding_mask if key_padding_mask is not None else attn_mask,
            #         need_weights,
            #         average_attn_weights,
            #         1 if key_padding_mask is not None else 0 if attn_mask is not None else None)

        # any_nested = query.is_nested or key.is_nested or value.is_nested
        # assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
        #                         f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # TODO: not adapted yet
            raise NotImplementedError

            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        in_proj_weight = [self.q_proj_weight, self.k_proj_weight, self.v_proj_weight]
        in_proj_bias = [self.q_in_proj_bias, self.k_in_proj_bias, self.v_in_proj_bias]
        in_proj_weight_teacher = [self.teacher_q_proj_weight, self.teacher_k_proj_weight, self.teacher_v_proj_weight]
        in_proj_bias_teacher = [self.teacher_q_in_proj_bias, self.teacher_k_in_proj_bias, self.teacher_v_in_proj_bias]

        attn_output, attn_output_weights, attn_output_weights_logits = multi_head_attention_forward(
            query, key, value, query_teacher, key_teacher, value_teacher, self.embed_dim, self.num_heads,
            in_proj_weight=in_proj_weight, in_proj_bias=in_proj_bias,
            in_proj_weight_teacher=in_proj_weight_teacher, in_proj_bias_teacher=in_proj_bias_teacher,
            bias_k=self.bias_k, bias_v=self.bias_v, add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias,
            out_proj_weight_teacher=self.teacher_out_proj.weight if self.teacher_out_proj is not None else None,
            out_proj_bias_teacher=self.teacher_out_proj.bias if self.teacher_out_proj is not None else None,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights,

            # use_separate_proj_weight=True,
            # q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
            # v_proj_weight=self.v_proj_weight,

            share_q=self.share_q,
            share_k=self.share_k,
            share_v=self.share_v,
            share_attn_map=self.share_attn_map,
            share_out_proj_weight=self.share_out_proj_weight,
        )

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights, attn_output_weights_logits
        else:
            return attn_output, attn_output_weights, attn_output_weights_logits


# def _linear(freeze_, q_, w_q_, b_q_):
#     if freeze_:
#         # add (w_q * 0 + b_q * 0) to use w_q, b_q in the graph but no grad will propagate back (so they will
#         # not be updated forever.
#         Q_ = linear(q_, w_q_.detach(), b_q_.detach()) + (torch.matmul(torch.zeros_like(q_), w_q_) + b_q_ * 0)
#     else:
#         Q_ = linear(q_, w_q_, b_q_)
#     return Q_


def _in_projection(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w_q: Tensor,
        w_k: Tensor,
        w_v: Tensor,
        q_teacher: Optional[Tensor] = None,
        k_teacher: Optional[Tensor] = None,
        v_teacher: Optional[Tensor] = None,
        w_q_teacher: Optional[Tensor] = None,
        w_k_teacher: Optional[Tensor] = None,
        w_v_teacher: Optional[Tensor] = None,
        b_q: Optional[Tensor] = None,
        b_k: Optional[Tensor] = None,
        b_v: Optional[Tensor] = None,
        b_q_teacher: Optional[Tensor] = None,
        b_k_teacher: Optional[Tensor] = None,
        b_v_teacher: Optional[Tensor] = None,
):  # -> Tuple[Tensor, Tensor, Tensor]:  -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    Ek_teacher = k_teacher.size(-1)
    Eq_teacher = q_teacher.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert w_q_teacher.shape == (Eq, Eq_teacher), f"expecting teacher key weights shape of {(Eq, Eq_teacher)}, " \
                                                  f"but got {w_q_teacher.shape}"
    assert w_k_teacher.shape == (Eq, Ek_teacher), f"expecting teacher key weights shape of {(Eq, Ek_teacher)}, " \
                                                  f"but got {w_k_teacher.shape}"

    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    assert b_q_teacher is None or b_q_teacher.shape == (Eq,), \
        f"expecting teacher key bias shape of {(Eq,)}, but got {b_q_teacher.shape}"
    assert b_k_teacher is None or b_k_teacher.shape == (Eq,), \
        f"expecting teacher key bias shape of {(Eq,)}, but got {b_k_teacher.shape}"

    # return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v), \
    #        linear(q_teacher, w_q_teacher, b_q_teacher), linear(k_teacher, w_k_teacher, b_k_teacher)
    Q = _linear(q, w_q, b_q)
    K = _linear(k, w_k, b_k)
    V = _linear(v, w_v, b_v)
    Q_teacher = _linear(q_teacher, w_q_teacher, b_q_teacher)
    K_teacher = _linear(k_teacher, w_k_teacher, b_k_teacher)
    V_teacher = _linear(v_teacher, w_v_teacher, b_v_teacher)
    return Q, K, V, Q_teacher, K_teacher, V_teacher


def multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_teacher: Tensor,
        key_teacher: Tensor,
        value_teacher: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: list,  # Optional[Tensor],
        in_proj_bias: list,  # Optional[Tensor],
        in_proj_weight_teacher: list,  # Optional[Tensor],
        in_proj_bias_teacher: list,  # Optional[Tensor],
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        out_proj_weight_teacher: Optional[Tensor] = None,
        out_proj_bias_teacher: Optional[Tensor] = None,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        # use_separate_proj_weight: bool = False,
        # q_proj_weight: Optional[Tensor] = None,
        # k_proj_weight: Optional[Tensor] = None,
        # v_proj_weight: Optional[Tensor] = None,
        # q_teacher_proj_weight: Optional[Tensor] = None,  # TODO: parent function not adapted to this variable yet.
        # k_teacher_proj_weight: Optional[Tensor] = None,  # TODO: parent function not adapted to this variable yet.
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        average_attn_weights: bool = True,

        share_q=False,
        share_k=False,
        share_v=False,
        share_attn_map=False,
        share_out_proj_weight=False,
):  # -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True
    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    # tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    # if has_torch_function(tens_ops):  # False,
    #     return handle_torch_function(
    #         multi_head_attention_forward,
    #         tens_ops,
    #         query,
    #         key,
    #         value,
    #         embed_dim_to_check,
    #         num_heads,
    #         in_proj_weight,
    #         in_proj_bias,
    #         bias_k,
    #         bias_v,
    #         add_zero_attn,
    #         dropout_p,
    #         out_proj_weight,
    #         out_proj_bias,
    #         training=training,
    #         key_padding_mask=key_padding_mask,
    #         need_weights=need_weights,
    #         attn_mask=attn_mask,
    #         use_separate_proj_weight=use_separate_proj_weight,
    #         q_proj_weight=q_proj_weight,
    #         k_proj_weight=k_proj_weight,
    #         v_proj_weight=v_proj_weight,
    #         static_k=static_k,
    #         static_v=static_v,
    #         average_attn_weights=average_attn_weights,
    #
    #         # pre_calculated_attn=pre_calculated_attn,
    #         # freeze_wq=freeze_wq,
    #         # freeze_wk=freeze_wk,
    #         # freeze_wv=freeze_wv,
    #     )

    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        raise NotImplementedError

        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    if key_padding_mask is not None:
        _kpm_dtype = key_padding_mask.dtype
        if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
            raise AssertionError(
                "only bool and floating types of key_padding_mask are supported")
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    # if use_separate_proj_weight:
    #     # allow MHA to have different embedding dimensions when separate projection weights are used
    #     assert key.shape[:2] == value.shape[:2], \
    #         f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    # else:
    #     assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
    assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
    #
    # compute in-projection
    #
    # if not use_separate_proj_weight:
    #     assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
    #     # q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    #     q, k, v, q_teacher, k_teacher = _in_projection_packed(query, key, value, query_teacher, key_teacher,
    #                                                           in_proj_weight, in_proj_bias,
    #                                                           freeze_wq=freeze_wq, freeze_wk=freeze_wk,
    #                                                           freeze_wv=freeze_wv,
    #                                                           freeze_wq_teacher=freeze_wq_teacher,
    #                                                           freeze_wk_teacher=freeze_wk_teacher
    #                                                           )
    # else:
    #     assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
    #     assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
    #     assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
    #     assert q_teacher_proj_weight is not None, "use_separate_proj_weight is True but q_teacher_proj_weight is None"
    #     assert k_teacher_proj_weight is not None, "use_separate_proj_weight is True but k_teacher_proj_weight is None"

    # in_proj_weight = [self.q_proj_weight, self.k_proj_weight, self.v_proj_weight]
    # in_proj_bias = [self.q_in_proj_bias, self.k_in_proj_bias, self.v_in_proj_bias]
    # in_proj_weight_teacher = [self.q_teacher_proj_weight, self.k_teacher_proj_weight, self.v_teacher_proj_weight]
    # in_proj_bias_teacher = [self.q_teacher_in_proj_bias, self.k_teacher_in_proj_bias, self.v_teacher_in_proj_bias]
    q_proj_weight, k_proj_weight, v_proj_weight = in_proj_weight
    b_q, b_k, b_v = in_proj_bias
    q_teacher_proj_weight, k_teacher_proj_weight, v_teacher_proj_weight = in_proj_weight_teacher
    b_q_teacher, b_k_teacher, b_v_teacher = in_proj_bias_teacher

    q, k, v, q_teacher, k_teacher, v_teacher = _in_projection(
        q=query, k=key, v=value, w_q=q_proj_weight, w_k=k_proj_weight, w_v=v_proj_weight,
        w_q_teacher=q_teacher_proj_weight, w_k_teacher=k_teacher_proj_weight, w_v_teacher=v_teacher_proj_weight,
        q_teacher=query_teacher, k_teacher=key_teacher, v_teacher=value_teacher,
        b_q=b_q, b_k=b_k, b_v=b_v, b_q_teacher=b_q_teacher, b_k_teacher=b_k_teacher, b_v_teacher=b_v_teacher,
    )

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])

        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None


    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k

    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)

        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)

    if attn_mask is not None:
        attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
    else:
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))

    attn_output_weights_logits = attn_output_weights.clone()
    attn_output_weights = softmax(attn_output_weights, dim=-1)  # torch.Size([16, 696, 696])

    if dropout_p > 0.0:
        attn_output_weights = dropout(attn_output_weights, p=dropout_p)

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    # --------------------------
    # second attention branch
    # --------------------------
    if share_attn_map:
        attn_output_weights_teacher = attn_output_weights
        attn_output_weights_logits_teacher = attn_output_weights_logits
    else:
        if share_q:
            assert q_teacher is None
            q_teacher_scaled = q_scaled
        else:
            assert q_teacher is not None
            q_teacher = q_teacher.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            q_teacher_scaled = q_teacher / math.sqrt(E)

        if share_k:
            assert k_teacher is None
            k_teacher = k
        else:
            assert k_teacher is not None

            if static_k is None:
                k_teacher = k_teacher.contiguous().view(k_teacher.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
            else:
                # TODO finish disentangling control flow so we don't do in-projections when statics are passed
                assert static_k.size(0) == bsz * num_heads, \
                    f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
                assert static_k.size(2) == head_dim, \
                    f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
                k_teacher = static_k

            # add zero attention along batch dimension (now first)
            if add_zero_attn:
                zero_attn_shape = (bsz * num_heads, 1, head_dim)

                k_teacher = torch.cat([k_teacher, torch.zeros(zero_attn_shape,
                                                              dtype=k_teacher.dtype, device=k_teacher.device)], dim=1)

        # ======================== share Q, V with backpropagation
        if attn_mask is not None:
            attn_output_weights_teacher = torch.baddbmm(attn_mask, q_teacher_scaled, k_teacher.transpose(-2, -1))
        else:
            attn_output_weights_teacher = torch.bmm(q_teacher_scaled, k_teacher.transpose(-2, -1))

        attn_output_weights_logits_teacher = attn_output_weights_teacher.clone()
        attn_output_weights_teacher = softmax(attn_output_weights_teacher, dim=-1)  # torch.Size([16, 696, 696])

        if dropout_p > 0.0:
            attn_output_weights_teacher = dropout(attn_output_weights_teacher, p=dropout_p)

    if share_v:
        assert v_teacher is None
        v_teacher = v
    else:
        assert v_teacher is not None

        if static_v is None:
            v_teacher = v_teacher.contiguous().view(v_teacher.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_v.size(0) == bsz * num_heads, \
                f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
            assert static_v.size(2) == head_dim, \
                f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
            v_teacher = static_v

        # add zero attention along batch dimension (now first)
        if add_zero_attn:
            zero_attn_shape = (bsz * num_heads, 1, head_dim)

            v_teacher = torch.cat([v_teacher, torch.zeros(zero_attn_shape,
                                                          dtype=v_teacher.dtype, device=v_teacher.device)], dim=1)

    attn_output_teacher = torch.bmm(attn_output_weights_teacher, v_teacher)

    attn_output_teacher = attn_output_teacher.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    if share_out_proj_weight:
        attn_output_teacher = linear(attn_output_teacher, out_proj_weight, out_proj_bias)
    else:
        attn_output_teacher = linear(attn_output_teacher, out_proj_weight_teacher, out_proj_bias_teacher)
    attn_output_teacher = attn_output_teacher.view(tgt_len, bsz, attn_output_teacher.size(1))

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights_teacher = attn_output_weights_teacher.view(bsz, num_heads, tgt_len, src_len)

        attn_output_weights_logits = attn_output_weights_logits.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights_logits_teacher = attn_output_weights_logits_teacher.view(bsz, num_heads, tgt_len, src_len)

        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
            attn_output_weights_teacher = attn_output_weights_teacher.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)

            attn_output_teacher = attn_output_teacher.squeeze(1)
            attn_output_weights_teacher = attn_output_weights_teacher.squeeze(0)
        return (attn_output, attn_output_teacher), (attn_output_weights, attn_output_weights_teacher), \
               (attn_output_weights_logits, attn_output_weights_logits_teacher)
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_teacher = attn_output_teacher.squeeze(1)
        return (attn_output, attn_output_teacher), None, None
