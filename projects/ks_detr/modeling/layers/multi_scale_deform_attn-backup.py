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
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/functions/ms_deform_attn_func.py
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/multi_scale_deform_attn.py
# ------------------------------------------------------------------------------------------------

import math
import warnings
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.init import constant_, xavier_uniform_

from detrex.layers.multi_scale_deform_attn import _is_power_of_2, create_dummy_class, \
    create_dummy_func, \
    MultiScaleDeformableAttnFunction, MultiScaleDeformableAttention
from .attention import with_pos_embed


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (
            value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        )
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


def get_query_and_value_teacher(query, query_pos, ksgt):
    assert ksgt is not None
    # TODO: replace this manually set q_teacher, v_teacher
    # query and query_pos: torch.Size([2, 11480, 256]), B, N, C, but ksgt_module requires x to be (N, B, C)
    ksgt_output = ksgt(x=query.permute(1, 0, 2),)  # return processed x: (N, B, C)
    src_with_gt = ksgt_output['x'].permute(1, 0, 2)  # change it to (B, N, C) again
    query_teacher = with_pos_embed(src_with_gt, pos=query_pos)  # k_teacher not used
    value_teacher = src_with_gt
    return query_teacher, value_teacher


class KSBaseMultiScaleDeformableAttention(MultiScaleDeformableAttention):
    """Multi-Scale Deformable Attention Module used in Deformable-DETR

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dim (int): The embedding dimension of Attention. Default: 256.
        num_heads (int): The number of attention heads. Default: 8.
        num_levels (int): The number of feature map used in Attention. Default: 4.
        num_points (int): The number of sampling points for each query
            in each head. Default: 4.
        img2col_steps (int): The step used in image_to_column. Defualt: 64.
            dropout (float): Dropout layer used in output. Default: 0.1.
        batch_first (bool): if ``True``, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """
    def forward_normal_attn(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            identity: Optional[torch.Tensor] = None,
            query_pos: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            reference_points: Optional[torch.Tensor] = None,
            spatial_shapes: Optional[torch.Tensor] = None,
            level_start_index: Optional[torch.Tensor] = None,
            **kwargs
        ):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(  # -4, 4
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )  # torch.Size([2, 10850, 256]) -> torch.Size([2, 10850, 8, 4, 4, 2])
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )  # torch.Size([2, 10850, 128]) -> torch.Size([2, 10850, 8, 16])  only 16 keys (n_lvl * n_point)
        # =======================
        attn_output_weight_logits = attention_weights.clone()
        # =======================

        attention_weights = attention_weights.softmax(-1)  # torch.Size([2, 10850, 8, 16])
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )  # torch.Size([2, 10850, 8, 4, 4])

        # bs, num_query, num_heads, num_levels, num_points, 2
        # spatial_shapes: tensor([[102,  80],   [ 51,  40],   [ 26,  20],  [ 13,  10]], device='cuda:0')
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                    reference_points[:, :, None, :, None, :]
                    + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                    reference_points[:, :, None, :, None, :2]
                    + sampling_offsets
                    / self.num_points
                    * reference_points[:, :, None, :, None, 2:]
                    * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )
        #
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:  # cpu version
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )
        output = self.output_proj(output)

        return output, value, sampling_offsets, attention_weights, attn_output_weight_logits, sampling_locations

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

    def forward_future_version(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            identity: Optional[torch.Tensor] = None,
            query_pos: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            reference_points: Optional[torch.Tensor] = None,
            spatial_shapes: Optional[torch.Tensor] = None,
            level_start_index: Optional[torch.Tensor] = None,
            **kwargs
        ):
        output, value, sampling_offsets, attention_weights, attn_output_weight_logits, sampling_locations = self.forward_normal_attn(
            query=query,
            key=key,
            value=value,
            identity=identity,
            query_pos=query_pos,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )
        output = self.identity_mapping(output=output, identity=identity)
        self_attn_out_dict = dict(attn_map_logits=attn_output_weight_logits)
        return output, self_attn_out_dict

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs
    ):

        """Forward Function of KSBaseMultiScaleDeformableAttention

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)`
            value (torch.Tensor): Value embeddings with shape
                `(num_key, bs, embed_dim)`
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default: None. If None, `query` will be
                used.
            query_pos (torch.Tensor): The position embedding for `query`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with shape `(bs, num_key)`,
                indicating which elements within `key` to be ignored in attention.
            reference_points (torch.Tensor): The normalized reference points
                with shape `(bs, num_query, num_levels, 2)`,
                all elements is range in [0, 1], top-left (0, 0),
                bottom-right (1, 1), including padding are.
                or `(N, Length_{query}, num_levels, 4)`, add additional
                two dimensions `(h, w)` to form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in different levels.
                With shape `(num_levels, 2)`, last dimension represents `(h, w)`.
            level_start_index (torch.Tensor): The start index of each level. A tensor with
                shape `(num_levels, )` which can be represented as
                `[0, h_0 * w_0, h_0 * w_0 + h_1 * w_1, ...]`.

        Returns:
            torch.Tensor: forward results with shape `(num_query, bs, embed_dim)`
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        # =========================================
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(  # -4, 4
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )  # torch.Size([2, 10850, 256]) -> torch.Size([2, 10850, 8, 4, 4, 2])
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )  # torch.Size([2, 10850, 128]) -> torch.Size([2, 10850, 8, 16])  only 16 keys (n_lvl * n_point)
        # =======================
        attn_output_weight_logits = attention_weights.clone()
        # =======================

        attention_weights = attention_weights.softmax(-1)  # torch.Size([2, 10850, 8, 16])
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )  # torch.Size([2, 10850, 8, 4, 4])

        # bs, num_query, num_heads, num_levels, num_points, 2
        # spatial_shapes: tensor([[102,  80],   [ 51,  40],   [ 26,  20],  [ 13,  10]], device='cuda:0')
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )
        #
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:  # cpu version
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )
        # ######################################
        output = self.output_proj(output)

        # if not self.batch_first:
        #     output = output.permute(1, 0, 2)
        # return self.dropout(output) + identity

        self_attn_out_dict = dict(attn_map_logits=attn_output_weight_logits)
        if torch.is_tensor(output):
            if not self.batch_first:
                output = output.permute(1, 0, 2)
            output = [identity + self.dropout(output)]
        else:
            for k, single_out in enumerate(output):
                if not self.batch_first:
                    single_out = single_out.permute(1, 0, 2)
                output[k] = identity + self.dropout(single_out)

        return output, self_attn_out_dict


class KSMultiScaleQuadAttentionShareOutProj(KSBaseMultiScaleDeformableAttention):

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        img2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = False,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            img2col_step=img2col_step,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.sampling_offsets_teacher = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights_teacher = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj_teacher = nn.Linear(embed_dim, embed_dim)
        # self.output_proj_teacher = nn.Linear(embed_dim, embed_dim)
        self.output_proj_teacher = None

        self.init_teacher_weights()

    def init_teacher_weights(self):
        """
        Default initialization for Parameters of the teacher branch.
        """
        if self.sampling_offsets_teacher is not None:
            constant_(self.sampling_offsets_teacher.weight.data, 0.0)

        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        if self.sampling_offsets_teacher is not None:
            with torch.no_grad():
                self.sampling_offsets_teacher.bias = nn.Parameter(grid_init.view(-1))

        if self.attention_weights_teacher is not None:
            constant_(self.attention_weights_teacher.weight.data, 0.0)
            constant_(self.attention_weights_teacher.bias.data, 0.0)

        if self.value_proj_teacher:
            xavier_uniform_(self.value_proj_teacher.weight.data)
            constant_(self.value_proj_teacher.bias.data, 0.0)

        if self.output_proj_teacher:
            xavier_uniform_(self.output_proj_teacher.weight.data)
            constant_(self.output_proj_teacher.bias.data, 0.0)

    def forward(
                self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs
        ):

        """Forward Function of KSBaseMultiScaleDeformableAttention

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)`
            value (torch.Tensor): Value embeddings with shape
                `(num_key, bs, embed_dim)`
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default: None. If None, `query` will be
                used.
            query_pos (torch.Tensor): The position embedding for `query`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with shape `(bs, num_key)`,
                indicating which elements within `key` to be ignored in attention.
            reference_points (torch.Tensor): The normalized reference points
                with shape `(bs, num_query, num_levels, 2)`,
                all elements is range in [0, 1], top-left (0, 0),
                bottom-right (1, 1), including padding are.
                or `(N, Length_{query}, num_levels, 4)`, add additional
                two dimensions `(h, w)` to form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in different levels.
                With shape `(num_levels, 2)`, last dimension represents `(h, w)`.
            level_start_index (torch.Tensor): The start index of each level. A tensor with
                shape `(num_levels, )` which can be represented as
                `[0, h_0 * w_0, h_0 * w_0 + h_1 * w_1, ...]`.

        Returns:
            torch.Tensor: forward results with shape `(num_query, bs, embed_dim)`
        """

        query_teacher, value_teacher = get_query_and_value_teacher(
            query=query, query_pos=query_pos, ksgt=kwargs['ksgt'])

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(  # -4, 4
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )  # torch.Size([2, 10850, 256]) -> torch.Size([2, 10850, 8, 4, 4, 2])
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )  # torch.Size([2, 10850, 128]) -> torch.Size([2, 10850, 8, 16])  only 16 keys (n_lvl * n_point)
        # =======================
        attn_output_weight_logits = attention_weights.clone()
        # =======================

        attention_weights = attention_weights.softmax(-1)  # torch.Size([2, 10850, 8, 16])
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )  # torch.Size([2, 10850, 8, 4, 4])

        # bs, num_query, num_heads, num_levels, num_points, 2
        # spatial_shapes: tensor([[102,  80],   [ 51,  40],   [ 26,  20],  [ 13,  10]], device='cuda:0')
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                    reference_points[:, :, None, :, None, :]
                    + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                    reference_points[:, :, None, :, None, :2]
                    + sampling_offsets
                    / self.num_points
                    * reference_points[:, :, None, :, None, 2:]
                    * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )
        #
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:  # cpu version
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )
        output = self.output_proj(output)

        # ######################################
        # Section 2: teacher branches
        # =========================================

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query_teacher = query_teacher.permute(1, 0, 2)
            value_teacher = value_teacher.permute(1, 0, 2)

        bs, num_query, _ = query_teacher.shape
        bs, num_value, _ = value_teacher.shape

        # assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        value_teacher = self.value_proj_teacher(value_teacher)

        if key_padding_mask is not None:
            # value = value.masked_fill(key_padding_mask[..., None], float(0))
            value_teacher = value_teacher.masked_fill(key_padding_mask[..., None], float(0))

        # value = value.view(bs, num_value, self.num_heads, -1)
        value_teacher = value_teacher.view(bs, num_value, self.num_heads, -1)

        sampling_offsets_teacher = self.sampling_offsets_teacher(query_teacher).view(  # -4, 4
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights_teacher = self.attention_weights_teacher(query_teacher).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )

        attn_output_weight_logits_teacher = attention_weights_teacher.clone()
        attention_weights_teacher = attention_weights_teacher.softmax(-1)  # torch.Size([2, 10850, 8, 16])
        attention_weights_teacher = attention_weights_teacher.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )  # torch.Size([2, 10850, 8, 4, 4])

        # bs, num_query, num_heads, num_levels, num_points, 2
        # spatial_shapes: tensor([[102,  80],   [ 51,  40],   [ 26,  20],  [ 13,  10]], device='cuda:0')
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations_teacher = (
                    reference_points[:, :, None, :, None, :]
                    + sampling_offsets_teacher / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations_teacher = (
                    reference_points[:, :, None, :, None, :2]
                    + sampling_offsets_teacher
                    / self.num_points
                    * reference_points[:, :, None, :, None, 2:]
                    * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )
        # attn_output_teacher1 = torch.bmm(attn_output_weights_teacher, v)
        # attn_output_teacher2 = torch.bmm(attn_output_weights, v_teacher)
        if torch.cuda.is_available() and value_teacher.is_cuda:
            output_teacher1 = MultiScaleDeformableAttnFunction.apply(  # Improve value
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations_teacher,
                attention_weights_teacher,
                self.im2col_step,
            )
            output_teacher2 = MultiScaleDeformableAttnFunction.apply(  # Improve weights
                value_teacher,
                spatial_shapes,
                level_start_index,
                sampling_locations_teacher,
                attention_weights,
                self.im2col_step,
            )
            output_teacher3 = MultiScaleDeformableAttnFunction.apply(  # Improve sampling locations
                value_teacher,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights_teacher,
                self.im2col_step,
            )
        else:  # cpu version
            output_teacher1 = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations_teacher, attention_weights_teacher
            )
            output_teacher2 = multi_scale_deformable_attn_pytorch(
                value_teacher, spatial_shapes, sampling_locations_teacher, attention_weights
            )
            output_teacher3 = multi_scale_deformable_attn_pytorch(
                value_teacher, spatial_shapes, sampling_locations, attention_weights_teacher
            )
        # ######################################
        output_teacher1 = self.output_proj(output_teacher1)
        output_teacher2 = self.output_proj(output_teacher2)
        output_teacher3 = self.output_proj(output_teacher3)

        output = [output] + [output_teacher1, output_teacher2, output_teacher3]
        # if not self.batch_first:
        #     output = output.permute(1, 0, 2)
        # return self.dropout(output) + identity

        self_attn_out_dict = dict(
            attn_map_logits=attn_output_weight_logits,
            attn_output_weight_logits_teacher=attn_output_weight_logits_teacher,
            sampling_offsets_teacher=sampling_offsets_teacher,
        )

        output = self.identity_mapping(output=output, identity=identity)

        return output, self_attn_out_dict


class KSMultiScaleTripleAttentionShareOutProj(KSBaseMultiScaleDeformableAttention):
    def __init__(
            self,
            embed_dim: int = 256,
            num_heads: int = 8,
            num_levels: int = 4,
            num_points: int = 4,
            img2col_step: int = 64,
            dropout: float = 0.1,
            batch_first: bool = False,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            img2col_step=img2col_step,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.attention_weights_teacher = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj_teacher = nn.Linear(embed_dim, embed_dim)
        # self.output_proj_teacher = nn.Linear(embed_dim, embed_dim)
        # self.sampling_offsets_teacher = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.output_proj_teacher = None
        self.sampling_offsets_teacher = None

        self.init_teacher_weights()

    def init_teacher_weights(self):
        """
        Default initialization for Parameters of the teacher branch.
        """
        if self.sampling_offsets_teacher is not None:
            constant_(self.sampling_offsets_teacher.weight.data, 0.0)

        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
                2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(self.num_heads, 1, 1, 2)
                .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        if self.sampling_offsets_teacher is not None:
            with torch.no_grad():
                self.sampling_offsets_teacher.bias = nn.Parameter(grid_init.view(-1))

        if self.attention_weights_teacher is not None:
            constant_(self.attention_weights_teacher.weight.data, 0.0)
            constant_(self.attention_weights_teacher.bias.data, 0.0)

        if self.value_proj_teacher:
            xavier_uniform_(self.value_proj_teacher.weight.data)
            constant_(self.value_proj_teacher.bias.data, 0.0)

        if self.output_proj_teacher:
            xavier_uniform_(self.output_proj_teacher.weight.data)
            constant_(self.output_proj_teacher.bias.data, 0.0)

    def forward(
            self,
            query: torch.Tensor,  # torch.Size([2, 16500, 256])
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            identity: Optional[torch.Tensor] = None,
            query_pos: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            reference_points: Optional[torch.Tensor] = None,
            spatial_shapes: Optional[torch.Tensor] = None,
            level_start_index: Optional[torch.Tensor] = None,
            **kwargs
    ):
        # ============================
        # We should use the initial query, not query + query_pos to get the query_teacher and value_teacher
        query_teacher, value_teacher = get_query_and_value_teacher(
            query=query, query_pos=query_pos, ksgt=kwargs['ksgt'])
        # ============================

        """Forward Function of KSBaseMultiScaleDeformableAttention

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)`
            value (torch.Tensor): Value embeddings with shape
                `(num_key, bs, embed_dim)`
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default: None. If None, `query` will be
                used.
            query_pos (torch.Tensor): The position embedding for `query`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with shape `(bs, num_key)`,
                indicating which elements within `key` to be ignored in attention.
            reference_points (torch.Tensor): The normalized reference points
                with shape `(bs, num_query, num_levels, 2)`,
                all elements is range in [0, 1], top-left (0, 0),
                bottom-right (1, 1), including padding are.
                or `(N, Length_{query}, num_levels, 4)`, add additional
                two dimensions `(h, w)` to form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in different levels.
                With shape `(num_levels, 2)`, last dimension represents `(h, w)`.
            level_start_index (torch.Tensor): The start index of each level. A tensor with
                shape `(num_levels, )` which can be represented as
                `[0, h_0 * w_0, h_0 * w_0 + h_1 * w_1, ...]`.

        Returns:
            torch.Tensor: forward results with shape `(num_query, bs, embed_dim)`
        """
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:  # torch.Size([2, 16500, 256])
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(  # -4, 4
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )  # torch.Size([2, 10850, 256]) -> torch.Size([2, 10850, 8, 4, 4, 2])
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )  # torch.Size([2, 10850, 128]) -> torch.Size([2, 10850, 8, 16])  only 16 keys (n_lvl * n_point)
        # =======================
        attn_output_weight_logits = attention_weights.clone()
        # =======================

        attention_weights = attention_weights.softmax(-1)  # torch.Size([2, 10850, 8, 16])
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )  # torch.Size([2, 10850, 8, 4, 4])

        # bs, num_query, num_heads, num_levels, num_points, 2
        # spatial_shapes: tensor([[102,  80],   [ 51,  40],   [ 26,  20],  [ 13,  10]], device='cuda:0')
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                    reference_points[:, :, None, :, None, :]
                    + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                    reference_points[:, :, None, :, None, :2]
                    + sampling_offsets
                    / self.num_points
                    * reference_points[:, :, None, :, None, 2:]
                    * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )
        #
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:  # cpu version
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )
        output = self.output_proj(output)

        # ######################################
        # Section 2: teacher branches
        # =========================================

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query_teacher = query_teacher.permute(1, 0, 2)
            value_teacher = value_teacher.permute(1, 0, 2)

        bs, num_query, _ = query_teacher.shape
        bs, num_value, _ = value_teacher.shape

        # assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        value_teacher = self.value_proj_teacher(value_teacher)

        if key_padding_mask is not None:
            # value = value.masked_fill(key_padding_mask[..., None], float(0))
            value_teacher = value_teacher.masked_fill(key_padding_mask[..., None], float(0))

        # value = value.view(bs, num_value, self.num_heads, -1)
        value_teacher = value_teacher.view(bs, num_value, self.num_heads, -1)

        # sampling_offsets_teacher = self.sampling_offsets_teacher(query_teacher).view(  # -4, 4
        #     bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        # )
        attention_weights_teacher = self.attention_weights_teacher(query_teacher).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )

        attn_output_weight_logits_teacher = attention_weights_teacher.clone()
        attention_weights_teacher = attention_weights_teacher.softmax(-1)  # torch.Size([2, 10850, 8, 16])
        attention_weights_teacher = attention_weights_teacher.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )  # torch.Size([2, 10850, 8, 4, 4])

        # # bs, num_query, num_heads, num_levels, num_points, 2
        # # spatial_shapes: tensor([[102,  80],   [ 51,  40],   [ 26,  20],  [ 13,  10]], device='cuda:0')
        # if reference_points.shape[-1] == 2:
        #     offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        #     sampling_locations_teacher = (
        #             reference_points[:, :, None, :, None, :]
        #             + sampling_offsets_teacher / offset_normalizer[None, None, None, :, None, :]
        #     )
        # elif reference_points.shape[-1] == 4:
        #     sampling_locations_teacher = (
        #             reference_points[:, :, None, :, None, :2]
        #             + sampling_offsets_teacher
        #             / self.num_points
        #             * reference_points[:, :, None, :, None, 2:]
        #             * 0.5
        #     )
        # else:
        #     raise ValueError(
        #         "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
        #             reference_points.shape[-1]
        #         )
        #     )
        # attn_output_teacher1 = torch.bmm(attn_output_weights_teacher, v)
        # attn_output_teacher2 = torch.bmm(attn_output_weights, v_teacher)
        if torch.cuda.is_available() and value_teacher.is_cuda:
            output_teacher1 = MultiScaleDeformableAttnFunction.apply(  # Improve value
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights_teacher,
                self.im2col_step,
            )
            output_teacher2 = MultiScaleDeformableAttnFunction.apply(  # Improve weights
                value_teacher,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )

        else:  # cpu version
            output_teacher1 = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights_teacher
            )
            output_teacher2 = multi_scale_deformable_attn_pytorch(
                value_teacher, spatial_shapes, sampling_locations, attention_weights
            )
        # ######################################
        output_teacher1 = self.output_proj(output_teacher1)  # torch.Size([2, 16320, 256])
        output_teacher2 = self.output_proj(output_teacher2)

        output = [output] + [output_teacher1, output_teacher2]
        # if not self.batch_first:
        #     output = output.permute(1, 0, 2)
        # return self.dropout(output) + identity

        self_attn_out_dict = dict(
            attn_map_logits=attn_output_weight_logits,
            attn_output_weight_logits_teacher=attn_output_weight_logits_teacher,
            # sampling_offsets_teacher=sampling_offsets_teacher,
        )

        output = self.identity_mapping(output=output, identity=identity)

        return output, self_attn_out_dict

    # def haha(self):
    #     # ######################################
    #     # Section 2: teacher branches
    #     # =========================================
    #     ksgt = kwargs.get('ksgt', None)
    #     assert ksgt is not None
    #
    #     # TODO: replace this manually set q_teacher, v_teacher
    #     ksgt_output = ksgt(x=query, )  # TODO: Check if the mask is correct
    #     src_with_gt = ksgt_output['x']
    #     query_teacher = with_pos_embed(src_with_gt, pos=query_pos)  # k_teacher not used
    #     value_teacher = src_with_gt
    #     # =========================================
    #     if not self.batch_first:
    #         # change to (bs, num_query ,embed_dims)
    #         query_teacher = query_teacher.permute(1, 0, 2)
    #         value_teacher = value_teacher.permute(1, 0, 2)
    #
    #     # bs, num_query, _ = query.shape
    #     # bs, num_value, _ = value.shape
    #
    #     # assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
    #     value_teacher = self.value_proj_teacher(value_teacher)
    #
    #     if key_padding_mask is not None:
    #         # value = value.masked_fill(key_padding_mask[..., None], float(0))
    #         value_teacher = value_teacher.masked_fill(key_padding_mask[..., None], float(0))
    #
    #     # value = value.view(bs, num_value, self.num_heads, -1)
    #     value_teacher = value_teacher.view(bs, num_value, self.num_heads, -1)
    #
    #     # sampling_offsets_teacher = self.sampling_offsets_teacher(query_teacher).view(  # -4, 4
    #     #     bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
    #     # )
    #     attention_weights_teacher = self.attention_weights_teacher(query_teacher).view(
    #         bs, num_query, self.num_heads, self.num_levels * self.num_points
    #     )
    #
    #     # =======================
    #     attn_output_weight_logits_teacher = attention_weights_teacher.clone()
    #     # =======================
    #
    #     attention_weights_teacher = attention_weights_teacher.softmax(-1)  # torch.Size([2, 10850, 8, 16])
    #     attention_weights_teacher = attention_weights_teacher.view(
    #         bs,
    #         num_query,
    #         self.num_heads,
    #         self.num_levels,
    #         self.num_points,
    #     )  # torch.Size([2, 10850, 8, 4, 4])
    #
    #     # bs, num_query, num_heads, num_levels, num_points, 2
    #     # spatial_shapes: tensor([[102,  80],   [ 51,  40],   [ 26,  20],  [ 13,  10]], device='cuda:0')
    #     if reference_points.shape[-1] == 2:
    #         offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
    #         sampling_locations_teacher = (
    #                 reference_points[:, :, None, :, None, :]
    #                 + sampling_offsets_teacher / offset_normalizer[None, None, None, :, None, :]
    #         )
    #     elif reference_points.shape[-1] == 4:
    #         sampling_locations_teacher = (
    #                 reference_points[:, :, None, :, None, :2]
    #                 + sampling_offsets_teacher
    #                 / self.num_points
    #                 * reference_points[:, :, None, :, None, 2:]
    #                 * 0.5
    #         )
    #     else:
    #         raise ValueError(
    #             "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
    #                 reference_points.shape[-1]
    #             )
    #         )
    #     # attn_output_teacher1 = torch.bmm(attn_output_weights_teacher, v)
    #     # attn_output_teacher2 = torch.bmm(attn_output_weights, v_teacher)
    #     if torch.cuda.is_available() and value_teacher.is_cuda:
    #         output_teacher1 = MultiScaleDeformableAttnFunction.apply(  # Improve value
    #             value,
    #             spatial_shapes,
    #             level_start_index,
    #             sampling_locations_teacher,
    #             attention_weights_teacher,
    #             self.im2col_step,
    #         )
    #         output_teacher2 = MultiScaleDeformableAttnFunction.apply(  # Improve weights
    #             value_teacher,
    #             spatial_shapes,
    #             level_start_index,
    #             sampling_locations_teacher,
    #             attention_weights,
    #             self.im2col_step,
    #         )
    #         output_teacher3 = MultiScaleDeformableAttnFunction.apply( # Improve sampling locations
    #             value_teacher,
    #             spatial_shapes,
    #             level_start_index,
    #             sampling_locations,
    #             attention_weights_teacher,
    #             self.im2col_step,
    #         )
    #     else:  # cpu version
    #         output_teacher1 = multi_scale_deformable_attn_pytorch(
    #             value, spatial_shapes, sampling_locations_teacher, attention_weights_teacher
    #         )
    #         output_teacher2 = multi_scale_deformable_attn_pytorch(
    #             value_teacher, spatial_shapes, sampling_locations_teacher, attention_weights
    #         )
    #         output_teacher3 = multi_scale_deformable_attn_pytorch(
    #             value_teacher, spatial_shapes, sampling_locations, attention_weights_teacher
    #         )
    #     # ######################################
    #     output_teacher1 = self.output_proj(output_teacher1)
    #     output_teacher2 = self.output_proj(output_teacher2)
    #     output_teacher3 = self.output_proj(output_teacher3)

try:
    from detrex import _C
except ImportError:
    # TODO: register ops natively so there is no need to import _C.
    _msg = "detrex is not compiled successfully, please build following the instructions!"
    _args = ("detrex._C", _msg)
    KSBaseMultiScaleDeformableAttention = create_dummy_class(  # noqa
        "KSBaseMultiScaleDeformableAttention", *_args
    )
