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

import copy
import warnings
from typing import List
import torch
import torch.nn as nn
from torch import nn as nn

from detrex.layers import BaseTransformerLayer, FFN, ConditionalSelfAttention, ConditionalCrossAttention, \
    MultiheadAttention, MultiScaleDeformableAttention
from .attention import (
    KSBaseMultiheadAttention,
    KSBaseMultiheadAttentionSeparateWeight,
    KSMultiheadAttentionWithGT,
    # KSBaseMultiheadDualAttention,
    KSMultiheadDualAttentionShareVOutProjV0,
    KSMultiheadDualAttentionShareAttnOutProjV0,
    KSMultiheadDualAttentionShareVOutProj,
    KSMultiheadDualAttentionShareAttnOutProj,
    KSMultiheadTripleAttentionQKVShareAttnOutProjV0,
    # KSConditionalSelfAttention,
    # KSConditionalCrossAttention
)
from .multi_scale_deform_attn import (
    KSBaseMultiScaleDeformableAttention,
    KSMultiScaleQuadAttentionShareOutProj,
    KSMultiScaleTripleAttentionShareOutProj,
    KSMultiScaleTripleAttentionShareOutProjV0,
)

from projects.group_detr.modeling.attention import GroupConditionalSelfAttention

from projects.ks_detr.modeling.ks_utils import parser_encoder_decoder_layers


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# Custom attention modules defined for this project
# "self_attn" should start with "self_attn", "cross_attn" should start with "cross_attn",
AdditionalAttentionSet = {}


def is_valid_attention_definition(attn_set: dict):
    if len(attn_set) == 0:
        return True
    else:
        default_attn = {"self_attn", "cross_attn"}
        assert default_attn.intersection(attn_set) == 0, \
            f'there should be no overlap for the operation list, but  default_attn = {default_attn},\n' \
            f' AdditionalAttentionSet={attn_set}'

        # "self_attn" should start with "self_attn", "cross_attn" should start with "cross_attn",
        for attn_name in attn_set:
            assert attn_name.startswith("self_attn") or attn_name.startswith("cross_attn")

        return True


class KSBaseTransformerEncoderLayer(nn.Module):
    # TODO: add more tutorials about BaseTransformerLayer
    """The implementation of Base `TransformerLayer` used in Transformer. Modified
    from `mmcv <https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py>`_.

    It can be built by directly passing the `Attentions`, `FFNs`, `Norms`
    module, which support more flexible cusomization combined with
    `LazyConfig` system. The `BaseTransformerLayer` also supports `prenorm`
    when you specifying the `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn (list[nn.Module] | nn.Module): nn.Module or a list
            contains the attention module used in TransformerLayer.
        ffn (nn.Module): FFN module used in TransformerLayer.
        norm (nn.Module): Normalization layer used in TransformerLayer.
        operation_order (tuple[str]): The execution order of operation in
            transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying the first element as `norm`.
            Default = None.
    """

    def __init__(
        self,
        attn: List[nn.Module],
        ffn: nn.Module,
        norm: nn.Module,
        operation_order: tuple = None,
    ):
        super(KSBaseTransformerEncoderLayer, self).__init__()

        # check if the defined attn valid
        assert is_valid_attention_definition(attn_set=AdditionalAttentionSet)

        default_operation = {"self_attn", "norm", "cross_attn", "ffn"}
        assert set(operation_order).issubset(
            default_operation.union(AdditionalAttentionSet)
        )

        # count attention nums
        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")
        # count attention numbers for additional attention modules.
        for attn_name in AdditionalAttentionSet:
            num_attn += operation_order.count(attn_name)

        if isinstance(attn, nn.Module):  # one attention module in a layer
            attn = [copy.deepcopy(attn) for _ in range(num_attn)]
        else:  # multiple attention modules (e.g., self-attn, cross-attn) in a layer
            assert len(attn) == num_attn, (
                f"The length of attn (nn.Module or List[nn.Module]) {num_attn}"
                f"is not consistent with the number of attention in "
                f"operation_order {operation_order}"
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = nn.ModuleList()
        index = 0
        for operation_name in operation_order:
            # add additional attention list
            if operation_name in ["self_attn", "cross_attn"] + list(AdditionalAttentionSet):
                self.attentions.append(attn[index])
                index += 1

        self.embed_dim = self.attentions[0].embed_dim

        # count ffn nums
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count("ffn")
        for _ in range(num_ffns):
            self.ffns.append(copy.deepcopy(ffn))

        # count norm nums
        self.norms = nn.ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(copy.deepcopy(norm))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        key_pos: torch.Tensor = None,
        attn_masks: List[torch.Tensor] = None,
        query_key_padding_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        **kwargs,
    ):
        """Forward function for `BaseTransformerLayer`.

        **kwargs contains the specific arguments of attentions.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
                which should be specified follows the attention module used in
                `BaseTransformerLayer`.
            key (torch.Tensor): Key embeddings used in `Attention`.
            value (torch.Tensor): Value embeddings with the same shape as `key`.
            query_pos (torch.Tensor): The position embedding for `query`.
                Default: None.
            key_pos (torch.Tensor): The position embedding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): A list of 2D ByteTensor used
                in calculation the corresponding attention. The length of
                `attn_masks` should be equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape `(bs, num_query)`. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (torch.Tensor): ByteTensor for `key`, with
                shape `(bs, num_key)`. Default: None.
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        intermediate_output = {}

        for layer in self.operation_order:
            # if layer == "self_attn":
            if layer.startswith("self_attn"):
                temp_key = temp_value = query  # torch.Size([2, 10850, 256])
                query, *self_attn_out = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1

                assert isinstance(query, list), 'returned query should be a list'
                identity = query

                # update intermediate_output
                if self_attn_out:
                    assert isinstance(self_attn_out[0], dict)
                    intermediate_output.update(self_attn_out[0])

                # if torch.is_tensor(query):
                #     identity = query
                # else:
                #     assert isinstance(query, list)
                #     # Keep the first location for the student use.
                #     identity = query[0]

            elif layer == "norm":
                # check identity is also changed after the query is changed:
                # identity == query is True after 'norm' operation -> pass
                for k in range(len(query)):
                    query[k] = self.norms[norm_index](query[k])

                norm_index += 1

            # # elif layer == "cross_attn":
            # elif layer.startswith("cross_attn"):
            #     # TODO: adapt cross attn output
            #     for k in range(len(query)):
            #         query[k], *cross_attn_out = self.attentions[attn_index](
            #             query[k],
            #             key,
            #             value,
            #             identity[k] if self.pre_norm else None,
            #             query_pos=query_pos,
            #             key_pos=key_pos,
            #             attn_mask=attn_masks[attn_index],
            #             key_padding_mask=key_padding_mask,
            #             **kwargs,
            #         )
            #         # update intermediate_output with cross_attn_out
            #         if cross_attn_out:
            #             assert isinstance(cross_attn_out[0], dict)
            #             intermediate_output.update(cross_attn_out[0])
            #
            #     attn_index += 1
            #     identity = query

            elif layer == "ffn":
                for k in range(len(query)):
                    query[k] = self.ffns[ffn_index](query[k], identity if self.pre_norm else None)
                ffn_index += 1

        assert isinstance(query, list)
        intermediate_output['feat'] = query[0]

        if len(query) == 2:
            intermediate_output['feat_t'] = query[1]
        elif len(query) > 2:
            intermediate_output['feat_t'] = query[1:]

        query = query[0]

        return query, intermediate_output


# class KSBaseTransformerEncoderLayerWithGT(KSBaseTransformerEncoderLayer):
#
#     def forward(
#         self,
#         query: torch.Tensor,
#         key: torch.Tensor = None,
#         value: torch.Tensor = None,
#         query_pos: torch.Tensor = None,
#         key_pos: torch.Tensor = None,
#         attn_masks: List[torch.Tensor] = None,
#         query_key_padding_mask: torch.Tensor = None,
#         key_padding_mask: torch.Tensor = None,
#         **kwargs,
#     ):
#         """Forward function for `BaseTransformerLayer`.
#
#         **kwargs contains the specific arguments of attentions.
#
#         Args:
#             query (torch.Tensor): Query embeddings with shape
#                 `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
#                 which should be specified follows the attention module used in
#                 `BaseTransformerLayer`.
#             key (torch.Tensor): Key embeddings used in `Attention`.
#             value (torch.Tensor): Value embeddings with the same shape as `key`.
#             query_pos (torch.Tensor): The position embedding for `query`.
#                 Default: None.
#             key_pos (torch.Tensor): The position embedding for `key`.
#                 Default: None.
#             attn_masks (List[Tensor] | None): A list of 2D ByteTensor used
#                 in calculation the corresponding attention. The length of
#                 `attn_masks` should be equal to the number of `attention` in
#                 `operation_order`. Default: None.
#             query_key_padding_mask (torch.Tensor): ByteTensor for `query`, with
#                 shape `(bs, num_query)`. Only used in `self_attn` layer.
#                 Defaults to None.
#             key_padding_mask (torch.Tensor): ByteTensor for `key`, with
#                 shape `(bs, num_key)`. Default: None.
#         """
#         norm_index = 0
#         attn_index = 0
#         ffn_index = 0
#         identity = query
#         if attn_masks is None:
#             attn_masks = [None for _ in range(self.num_attn)]
#         elif isinstance(attn_masks, torch.Tensor):
#             attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
#             warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
#         else:
#             assert len(attn_masks) == self.num_attn, (
#                 f"The length of "
#                 f"attn_masks {len(attn_masks)} must be equal "
#                 f"to the number of attention in "
#                 f"operation_order {self.num_attn}"
#             )
#
#         intermediate_output = {}
#
#         for layer in self.operation_order:
#             # if layer == "self_attn":
#             if layer.startswith("self_attn"):
#                 # temp_key = temp_value = query
#                 query, *self_attn_out = self.attentions[attn_index](
#                     query,
#                     # temp_key,
#                     # temp_value,
#                     identity if self.pre_norm else None,
#                     query_pos=query_pos,
#                     # key_pos=query_pos,
#                     attn_mask=attn_masks[attn_index],
#                     key_padding_mask=query_key_padding_mask,
#                     **kwargs,
#                 )
#                 attn_index += 1
#
#                 assert isinstance(query, list), 'returned query should be a list'
#                 identity = query
#
#                 # update intermediate_output
#                 if self_attn_out:
#                     assert isinstance(self_attn_out[0], dict)
#                     intermediate_output.update(self_attn_out[0])
#
#                 # if torch.is_tensor(query):
#                 #     identity = query
#                 # else:
#                 #     assert isinstance(query, list)
#                 #     # Keep the first location for the student use.
#                 #     identity = query[0]
#
#             elif layer == "norm":
#                 # check identity is also changed after the query is changed:
#                 # identity == query is True after 'norm' operation -> pass
#                 for k in range(len(query)):
#                     query[k] = self.norms[norm_index](query[k])
#
#                 norm_index += 1
#
#             # # elif layer == "cross_attn":
#             # elif layer.startswith("cross_attn"):
#             #     # TODO: adapt cross attn output
#             #     for k in range(len(query)):
#             #         query[k], *cross_attn_out = self.attentions[attn_index](
#             #             query[k],
#             #             key,
#             #             value,
#             #             identity[k] if self.pre_norm else None,
#             #             query_pos=query_pos,
#             #             key_pos=key_pos,
#             #             attn_mask=attn_masks[attn_index],
#             #             key_padding_mask=key_padding_mask,
#             #             **kwargs,
#             #         )
#             #         # update intermediate_output with cross_attn_out
#             #         if cross_attn_out:
#             #             assert isinstance(cross_attn_out[0], dict)
#             #             intermediate_output.update(cross_attn_out[0])
#             #
#             #     attn_index += 1
#             #     identity = query
#
#             elif layer == "ffn":
#                 for k in range(len(query)):
#                     query[k] = self.ffns[ffn_index](query[k], identity if self.pre_norm else None)
#                 ffn_index += 1
#
#         assert isinstance(query, list)
#         if len(query) == 1:
#             query = query[0]
#         else:
#             intermediate_output['feat'] = query[1:]
#             query = query[0]

#         return query, intermediate_output


# class KSBaseTransformerDecoderLayer(KSBaseTransformerEncoderLayer):
#
#     def forward(
#         self,
#         query: torch.Tensor,
#         key: torch.Tensor = None,
#         value: torch.Tensor = None,
#         query_pos: torch.Tensor = None,
#         key_pos: torch.Tensor = None,
#         attn_masks: List[torch.Tensor] = None,
#         query_key_padding_mask: torch.Tensor = None,
#         key_padding_mask: torch.Tensor = None,
#         **kwargs,
#     ):
#         """Forward function for `BaseTransformerLayer`.
#
#         **kwargs contains the specific arguments of attentions.
#
#         Args:
#             query (torch.Tensor): Query embeddings with shape
#                 `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
#                 which should be specified follows the attention module used in
#                 `BaseTransformerLayer`.
#             key (torch.Tensor): Key embeddings used in `Attention`.
#             value (torch.Tensor): Value embeddings with the same shape as `key`.
#             query_pos (torch.Tensor): The position embedding for `query`.
#                 Default: None.
#             key_pos (torch.Tensor): The position embedding for `key`.
#                 Default: None.
#             attn_masks (List[Tensor] | None): A list of 2D ByteTensor used
#                 in calculation the corresponding attention. The length of
#                 `attn_masks` should be equal to the number of `attention` in
#                 `operation_order`. Default: None.
#             query_key_padding_mask (torch.Tensor): ByteTensor for `query`, with
#                 shape `(bs, num_query)`. Only used in `self_attn` layer.
#                 Defaults to None.
#             key_padding_mask (torch.Tensor): ByteTensor for `key`, with
#                 shape `(bs, num_key)`. Default: None.
#         """
#         norm_index = 0
#         attn_index = 0
#         ffn_index = 0
#         identity = query
#         if attn_masks is None:
#             attn_masks = [None for _ in range(self.num_attn)]
#         elif isinstance(attn_masks, torch.Tensor):
#             attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
#             warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
#         else:
#             assert len(attn_masks) == self.num_attn, (
#                 f"The length of "
#                 f"attn_masks {len(attn_masks)} must be equal "
#                 f"to the number of attention in "
#                 f"operation_order {self.num_attn}"
#             )
#
#         intermediate_output = {}
#
#         for layer in self.operation_order:
#             # if layer == "self_attn":
#             if layer.startswith("self_attn"):
#                 temp_key = temp_value = query
#                 query, *self_attn_out = self.attentions[attn_index](
#                     query,
#                     temp_key,
#                     temp_value,
#                     identity if self.pre_norm else None,
#                     query_pos=query_pos,
#                     key_pos=query_pos,
#                     attn_mask=attn_masks[attn_index],
#                     key_padding_mask=query_key_padding_mask,
#                     **kwargs,
#                 )
#                 attn_index += 1
#
#                 assert isinstance(query, list), 'returned query should be a list'
#                 identity = query
#
#                 # update intermediate_output
#                 if self_attn_out:
#                     assert isinstance(self_attn_out[0], dict)
#                     intermediate_output.update(self_attn_out[0])
#
#                 # if torch.is_tensor(query):
#                 #     identity = query
#                 # else:
#                 #     assert isinstance(query, list)
#                 #     # Keep the first location for the student use.
#                 #     identity = query[0]
#
#             elif layer == "norm":
#                 # check identity is also changed after the query is changed:
#                 # identity == query is True after 'norm' operation -> pass
#                 for k in range(len(query)):
#                     query[k] = self.norms[norm_index](query[k])
#
#                 norm_index += 1
#
#             # elif layer == "cross_attn":
#             elif layer.startswith("cross_attn"):
#                 # TODO: adapt cross attn output
#                 for k in range(len(query)):
#                     query[k], *cross_attn_out = self.attentions[attn_index](
#                         query[k],
#                         key,
#                         value,
#                         identity[k] if self.pre_norm else None,
#                         query_pos=query_pos,
#                         key_pos=key_pos,
#                         attn_mask=attn_masks[attn_index],
#                         key_padding_mask=key_padding_mask,
#                         **kwargs,
#                     )
#                     # update intermediate_output with cross_attn_out
#                     if cross_attn_out:
#                         assert isinstance(cross_attn_out[0], dict)
#                         intermediate_output.update(cross_attn_out[0])
#
#                 attn_index += 1
#                 identity = query
#
#             elif layer == "ffn":
#                 for k in range(len(query)):
#                     query[k] = self.ffns[ffn_index](query[k], identity if self.pre_norm else None)
#                 ffn_index += 1
#
#         assert isinstance(query, list)
#         if len(query) == 1:
#             query = query[0]
#         else:
#             intermediate_output['feat'] = query[1:]
#             query = query[0]

#         return query, intermediate_output


class KSTransformerLayerSequence(nn.Module):
    """Base class for TransformerEncoder and TransformerDecoder, which will copy
    the passed `transformer_layers` module `num_layers` time or save the passed
    list of `transformer_layers` as parameters named ``self.layers``
    which is the type of ``nn.ModuleList``.
    The users should inherit `TransformerLayerSequence` and implemente their
    own forward function.

    Args:
        transformer_layers (list[KSBaseTransformerEncoderLayer] | KSBaseTransformerEncoderLayer): A list
            of BaseTransformerLayer. If it is obj:`BaseTransformerLayer`, it
            would be repeated `num_layers` times to a list[BaseTransformerLayer]
        num_layers (int): The number of `TransformerLayer`. Default: None.
    """

    def __init__(
        self,
        encoder_decoder_layer_list=None,

        transformer_layers=None,
        num_layers=None,
    ):
        super(KSTransformerLayerSequence, self).__init__()

        # Initialize layers first from encoder_layer_list, if encoder_layer_list is None,
        # then go to the pre-defined way of layer initialization.
        if encoder_decoder_layer_list is not None:
            self.layers = nn.ModuleList()
            self.num_layers = 0
            for l_conf in encoder_decoder_layer_list:
                encoder_layer, num_l = l_conf
                assert num_l > 0
                # nn.ModuleList
                self.layers.extend(_get_clones(encoder_layer, num_l))
                self.num_layers += num_l
        else:
            self.num_layers = num_layers
            self.layers = nn.ModuleList()
            if isinstance(transformer_layers, nn.Module):
                for _ in range(num_layers):
                    self.layers.append(copy.deepcopy(transformer_layers))
            else:
                assert isinstance(transformer_layers, list) and len(transformer_layers) == num_layers

    def forward(self):
        """Forward function of `TransformerLayerSequence`. The users should inherit
        `TransformerLayerSequence` and implemente their own forward function.
        """
        raise NotImplementedError()


EncoderLayerDict = {
    'regular': dict(
        LayerType=KSBaseTransformerEncoderLayer,
        SelfAttentionType=KSBaseMultiheadAttention,  # prepare Q, K, V and call attn func, conduct '+ identity'
    ),  # AttentionFunc=MultiheadAttention
    'regularSW': dict(  # DualAttnShareVOutProjFFN
        LayerType=KSBaseTransformerEncoderLayer,
        SelfAttentionType=KSBaseMultiheadAttentionSeparateWeight,
    ),
    'AttnWithGT': dict(LayerType=KSBaseTransformerEncoderLayer, SelfAttentionType=KSMultiheadAttentionWithGT, ),
    # 'FeatureWithGT': dict(
    #     LayerType=KSBaseTransformerEncoderLayerWithGT,
    #     SelfAttentionType=KSBaseMultiheadAttentionSeparateWeight,
    # ),
    'DualAttnShareVOutProjFFNV0': dict(LayerType=KSBaseTransformerEncoderLayer,
                                       SelfAttentionType=KSMultiheadDualAttentionShareVOutProjV0, ),
    'DualAttnShareVOutProjFFN': dict(LayerType=KSBaseTransformerEncoderLayer,
                                     SelfAttentionType=KSMultiheadDualAttentionShareVOutProj, ),
    'DualAttnShareAttnOutProjFFNV0': dict(LayerType=KSBaseTransformerEncoderLayer,
                                          SelfAttentionType=KSMultiheadDualAttentionShareAttnOutProjV0, ),
    'DualAttnShareAttnOutProjFFN': dict(LayerType=KSBaseTransformerEncoderLayer,
                                        SelfAttentionType=KSMultiheadDualAttentionShareAttnOutProj, ),
    'TripleAttnQKVShareAttnOutProjFFNV0': dict(LayerType=KSBaseTransformerEncoderLayer,
                                               SelfAttentionType=KSMultiheadTripleAttentionQKVShareAttnOutProjV0, ),
    # TransformerEncoderTripleAttnLayerShareQKVOutProjFFN,
    # 'DualAttnShareV': TransformerEncoderDualAttnLayerShareV,
    # 'DualAttnShareAttnFFN': TransformerEncoderDualAttnLayerShareAttnFFN,
    # 'DualAttnShareAttn': TransformerEncoderDualAttnLayerShareAttn,
}
DecoderLayerDict = {
    'regular': dict(LayerType=BaseTransformerLayer,  # KSBaseTransformerLayer
                    SelfAttentionType=None,
                    CrossAttentionType=None,  # TODO: adapt this
    ),
}


DeformableEncoderLayerDict = {
    'regular': dict(
        LayerType=KSBaseTransformerEncoderLayer,
        SelfAttentionType=KSBaseMultiScaleDeformableAttention,
        # prepare Q, K, V and call attn func, conduct '+ identity'
    ),  # AttentionFunc=MultiheadAttention
    # 'regularSW': dict(  # DualAttnShareVOutProjFFN
    #     LayerType=KSBaseTransformerEncoderLayer,
    #     SelfAttentionType=KSBaseMultiheadAttentionSeparateWeight,
    # ),
    # 'AttnWithGT': dict(LayerType=KSBaseTransformerEncoderLayer, SelfAttentionType=KSMultiheadAttentionWithGT, ),
    # # 'FeatureWithGT': dict(
    # #     LayerType=KSBaseTransformerEncoderLayerWithGT,
    # #     SelfAttentionType=KSBaseMultiheadAttentionSeparateWeight,
    # # ),
    # 'DualAttnShareVOutProjFFNV0': dict(LayerType=KSBaseTransformerEncoderLayer,
    #                                    SelfAttentionType=KSMultiScaleDeformableDualAttentionShareVOutProjV0, ),
    # 'DualAttnShareVOutProjFFN': dict(LayerType=KSBaseTransformerEncoderLayer,
    #                                  SelfAttentionType=KSMultiheadDualAttentionShareVOutProj, ),
    # 'DualAttnShareAttnOutProjFFNV0': dict(LayerType=KSBaseTransformerEncoderLayer,
    #                                       SelfAttentionType=KSMultiheadDualAttentionShareAttnOutProjV0, ),
    # 'DualAttnShareAttnOutProjFFN': dict(LayerType=KSBaseTransformerEncoderLayer,
    #                                     SelfAttentionType=KSMultiheadDualAttentionShareAttnOutProj, ),
    'DeformableTripleAttnShareAttnVOutProjFFN': dict(
        LayerType=KSBaseTransformerEncoderLayer,
        SelfAttentionType=KSMultiScaleTripleAttentionShareOutProj,
    ),
    'DeformableTripleAttnShareAttnVOutProjFFNV0': dict(
        LayerType=KSBaseTransformerEncoderLayer,
        SelfAttentionType=KSMultiScaleTripleAttentionShareOutProjV0,
    ),
    'DeformableQuadAttnShareAttnVOffsetOutProjFFN': dict(
        LayerType=KSBaseTransformerEncoderLayer,
        SelfAttentionType=KSMultiScaleQuadAttentionShareOutProj,
    ),
}

DeformableDecoderLayerDict = {
    'regular': dict(LayerType=BaseTransformerLayer,  # KSBaseTransformerLayer
                    SelfAttentionType=None,
                    CrossAttentionType=None,  # TODO: adapt this
    ),
}


def generate_transformer_encoder_layers(
        encoder_layer_dict, encoder_layer_config,
        embed_dim,
        num_heads,
        attn_dropout,
        feedforward_dim,
        ffn_dropout,
        activation,
        # post_norm: bool = False,
        batch_first,
):
    encoder_layer_conf_list = parser_encoder_decoder_layers(encoder_layer_config)

    encoder_layer_list = []
    for l_name, num_l in encoder_layer_conf_list:
        assert l_name in encoder_layer_dict and num_l > 0
        single_encoder_layer_config = encoder_layer_dict[l_name]
        encoder_layer = single_encoder_layer_config['LayerType'](  # KSBaseTransformerLayer
            attn=single_encoder_layer_config['SelfAttentionType'](  # KSMultiheadAttention
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_drop=attn_dropout,
                batch_first=batch_first,
                # self_attn_module=single_encoder_layer_config['AttentionFunc']
            ),
            ffn=FFN(
                embed_dim=embed_dim,
                feedforward_dim=feedforward_dim,
                ffn_drop=ffn_dropout,
                activation=activation,
            ),
            norm=nn.LayerNorm(normalized_shape=embed_dim),
            operation_order=("self_attn", "norm", "ffn", "norm"),
        )
        encoder_layer_list.append([encoder_layer, num_l])
    return encoder_layer_list


def generate_deformable_transformer_encoder_layers(
        encoder_layer_dict, encoder_layer_config,
        embed_dim,
        num_heads,
        attn_dropout,
        feedforward_dim,
        ffn_dropout,
        # activation,
        # post_norm: bool = False,
        num_feature_levels,
        operation_order: tuple = ("self_attn", "norm", "ffn", "norm"),
):
    encoder_layer_conf_list = parser_encoder_decoder_layers(encoder_layer_config)

    encoder_layer_list = []
    for l_name, num_l in encoder_layer_conf_list:
        assert l_name in encoder_layer_dict and num_l > 0
        single_encoder_layer_config = encoder_layer_dict[l_name]

        # transformer_layers = BaseTransformerLayer(
        #     attn=MultiScaleDeformableAttention(
        #         embed_dim=embed_dim,
        #         num_heads=num_heads,
        #         dropout=attn_dropout,
        #         batch_first=True,
        #         num_levels=num_feature_levels,
        #     ),
        #     ffn=FFN(
        #         embed_dim=embed_dim,
        #         feedforward_dim=feedforward_dim,
        #         output_dim=embed_dim,
        #         num_fcs=2,
        #         ffn_drop=ffn_dropout,
        #     ),
        #     norm=nn.LayerNorm(embed_dim),
        #     operation_order=("self_attn", "norm", "ffn", "norm"),
        # ),

        encoder_layer = single_encoder_layer_config['LayerType'](  # KSBaseTransformerLayer
            attn=single_encoder_layer_config['SelfAttentionType'](  # KSMultiheadAttention
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
                num_levels=num_feature_levels,
                # self_attn_module=single_encoder_layer_config['AttentionFunc']
            ),
            ffn=FFN(
                embed_dim=embed_dim,
                feedforward_dim=feedforward_dim,
                output_dim=embed_dim,
                num_fcs=2,
                ffn_drop=ffn_dropout,
            ),
            norm=nn.LayerNorm(embed_dim),
            operation_order=operation_order,
        )
        encoder_layer_list.append([encoder_layer, num_l])
    return encoder_layer_list


def generate_transformer_decoder_layers(
        decoder_layer_dict, decoder_layer_config,
        embed_dim,
        num_heads,
        attn_dropout,
        feedforward_dim,
        ffn_dropout,
        activation,
        # post_norm: bool = False,
        batch_first,
):
    """Currently only support conditional self attention and cross attention. """
    layer_conf_list = parser_encoder_decoder_layers(decoder_layer_config)

    layer_list = []
    for l_name, num_l in layer_conf_list:
        assert l_name in decoder_layer_dict and num_l > 0
        single_decoder_layer_config = decoder_layer_dict[l_name]

        decoder_layer = single_decoder_layer_config['LayerType'](  # KSBaseTransformerLayer
            attn=[
                ConditionalSelfAttention(  # TODO: update this
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    batch_first=batch_first,

                    # self_attn_module=single_decoder_layer_config['SelfAttentionType'],
                ),
                ConditionalCrossAttention(  # TODO: update this
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    batch_first=batch_first,
                    # cross_attn_module=single_decoder_layer_config['CrossAttentionType']
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
        )
        layer_list.append([decoder_layer, num_l])

    return layer_list


def generate_group_detr_transformer_decoder_layers(
        decoder_layer_dict, decoder_layer_config,
        embed_dim,
        num_heads,
        attn_dropout,
        feedforward_dim,
        ffn_dropout,
        activation,
        # post_norm: bool = False,
        batch_first,
        group_nums,
):
    """Currently only support conditional self attention and cross attention. """
    layer_conf_list = parser_encoder_decoder_layers(decoder_layer_config)

    layer_list = []
    for l_name, num_l in layer_conf_list:
        assert l_name in decoder_layer_dict and num_l > 0
        single_decoder_layer_config = decoder_layer_dict[l_name]

        """
            transformer_layers=BaseTransformerLayer(
                attn=[
                    GroupConditionalSelfAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        group_nums=group_nums,
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
        """
        decoder_layer = single_decoder_layer_config['LayerType'](  # KSBaseTransformerLayer
            attn=[
                GroupConditionalSelfAttention(  # TODO: update this
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    group_nums=group_nums,
                    batch_first=batch_first,
                    # self_attn_module=single_decoder_layer_config['SelfAttentionType'],
                ),
                ConditionalCrossAttention(  # TODO: update this
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    batch_first=batch_first,
                    # cross_attn_module=single_decoder_layer_config['CrossAttentionType']
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
        )
        layer_list.append([decoder_layer, num_l])

    return layer_list


def generate_deformable_transformer_decoder_layers(
        decoder_layer_dict, decoder_layer_config,
        embed_dim,
        num_heads,
        attn_dropout,
        feedforward_dim,
        ffn_dropout,
        num_feature_levels,
):
    layer_conf_list = parser_encoder_decoder_layers(decoder_layer_config)

    layer_list = []
    for l_name, num_l in layer_conf_list:
        assert l_name in decoder_layer_dict and num_l > 0
        single_decoder_layer_config = decoder_layer_dict[l_name]

        # transformer_layers = BaseTransformerLayer(
        #     attn=[
        #         MultiheadAttention(
        #             embed_dim=embed_dim,
        #             num_heads=num_heads,
        #             attn_drop=attn_dropout,
        #             batch_first=True,
        #         ),
        #         MultiScaleDeformableAttention(
        #             embed_dim=embed_dim,
        #             num_heads=num_heads,
        #             dropout=attn_dropout,
        #             batch_first=True,
        #             num_levels=num_feature_levels,
        #         ),
        #     ],
        #     ffn=FFN(
        #         embed_dim=embed_dim,
        #         feedforward_dim=feedforward_dim,
        #         output_dim=embed_dim,
        #         ffn_drop=ffn_dropout,
        #     ),
        #     norm=nn.LayerNorm(embed_dim),
        #     operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        # ),

        decoder_layer = single_decoder_layer_config['LayerType'](  # KSBaseTransformerLayer
            attn=[
                MultiheadAttention(  # TODO: update this
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    batch_first=True,
                    # self_attn_module=single_decoder_layer_config['SelfAttentionType'],
                ),
                MultiScaleDeformableAttention(  # TODO: update this
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                    num_levels=num_feature_levels,
                    # cross_attn_module=single_decoder_layer_config['CrossAttentionType']
                ),
            ],
            ffn=FFN(
                embed_dim=embed_dim,
                feedforward_dim=feedforward_dim,
                output_dim=embed_dim,
                ffn_drop=ffn_dropout,
            ),
            norm=nn.LayerNorm(embed_dim),
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        )
        layer_list.append([decoder_layer, num_l])
    return layer_list