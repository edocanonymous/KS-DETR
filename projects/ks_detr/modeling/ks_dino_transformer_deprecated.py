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
    # FFN,
    MLP,
    # BaseTransformerLayer,
    # MultiheadAttention,
    MultiScaleDeformableAttention,
    # TransformerLayerSequence,
    get_sine_pos_embed,
)
from detrex.utils import inverse_sigmoid

# ----------------------
from projects.ks_detr.modeling.layers import (
    KSTransformerLayerSequence,
    DeformableEncoderLayerDict,
    DeformableDecoderLayerDict,
    KSBaseMultiScaleDeformableAttention,
)
from projects.dino.modeling import DINOTransformer
from .detr_module_utils import (
    concat_student_teacher_decoder_output,
    extend_student_teacher_decoder_output,
    extract_optional_output,
)
from .layers.transformer import generate_deformable_transformer_encoder_layers, \
    generate_deformable_transformer_decoder_layers


# ----------------------


class KSDINOTransformerEncoder(KSTransformerLayerSequence):
    def __init__(
            self,
            embed_dim: int = 256,
            num_heads: int = 8,
            feedforward_dim: int = 1024,
            attn_dropout: float = 0.1,
            ffn_dropout: float = 0.1,
            # num_layers: int = 6,
            post_norm: bool = False,
            num_feature_levels: int = 4,

            encoder_layer_config: str = None,
    ):
        # =====================================-
        encoder_layer_list = generate_deformable_transformer_encoder_layers(
            encoder_layer_dict=DeformableEncoderLayerDict,
            encoder_layer_config=encoder_layer_config,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            feedforward_dim=feedforward_dim,
            ffn_dropout=ffn_dropout,
            num_feature_levels=num_feature_levels,
        )
        super().__init__(encoder_decoder_layer_list=encoder_layer_list)
        # =====================================-
        # super(KSDINOTransformerEncoder, self).__init__(
        #     transformer_layers=BaseTransformerLayer(
        #         attn=MultiScaleDeformableAttention(
        #             embed_dim=embed_dim,
        #             num_heads=num_heads,
        #             dropout=attn_dropout,
        #             batch_first=True,
        #             num_levels=num_feature_levels,
        #         ),
        #         ffn=FFN(
        #             embed_dim=embed_dim,
        #             feedforward_dim=feedforward_dim,
        #             output_dim=embed_dim,
        #             num_fcs=2,
        #             ffn_drop=ffn_dropout,
        #         ),
        #         norm=nn.LayerNorm(embed_dim),
        #         operation_order=("self_attn", "norm", "ffn", "norm"),
        #     ),
        #     num_layers=num_layers,
        # )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

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
        encoder_intermediate_output_list = []
        for layer in self.layers:
            query, *intermediate_output = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

            encoder_intermediate_output_list.append(extract_optional_output(intermediate_output))
        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query, encoder_intermediate_output_list


class KSDINOTransformerDecoder(KSTransformerLayerSequence):
    def __init__(
            self,
            embed_dim: int = 256,
            num_heads: int = 8,
            feedforward_dim: int = 1024,
            attn_dropout: float = 0.1,
            ffn_dropout: float = 0.1,
            # num_layers: int = 6,
            return_intermediate: bool = True,
            num_feature_levels: int = 4,
            look_forward_twice=True,

            decoder_layer_config: str = None,
    ):
        # --------------------
        decoder_layer_list = generate_deformable_transformer_decoder_layers(
            decoder_layer_dict=DeformableDecoderLayerDict,
            decoder_layer_config=decoder_layer_config,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            feedforward_dim=feedforward_dim,
            ffn_dropout=ffn_dropout,
            num_feature_levels=num_feature_levels,
        )
        super().__init__(encoder_decoder_layer_list=decoder_layer_list)
        num_layers = self.num_layers
        # --------------------
        # super(KSDINOTransformerDecoder, self).__init__(
        #     transformer_layers=BaseTransformerLayer(
        #         attn=[
        #             MultiheadAttention(
        #                 embed_dim=embed_dim,
        #                 num_heads=num_heads,
        #                 attn_drop=attn_dropout,
        #                 batch_first=True,
        #             ),
        #             MultiScaleDeformableAttention(
        #                 embed_dim=embed_dim,
        #                 num_heads=num_heads,
        #                 dropout=attn_dropout,
        #                 batch_first=True,
        #                 num_levels=num_feature_levels,
        #             ),
        #         ],
        #         ffn=FFN(
        #             embed_dim=embed_dim,
        #             feedforward_dim=feedforward_dim,
        #             output_dim=embed_dim,
        #             ffn_drop=ffn_dropout,
        #         ),
        #         norm=nn.LayerNorm(embed_dim),
        #         operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        #     ),
        #     num_layers=num_layers,
        # )
        self.return_intermediate = return_intermediate

        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)

        self.bbox_embed = None
        self.class_embed = None
        self.look_forward_twice = look_forward_twice
        self.norm = nn.LayerNorm(embed_dim)

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
            reference_points=None,  # num_queries, 4. normalized.
            valid_ratios=None,
            **kwargs,
    ):
        decoder_intermediate_out_list = []

        output = query
        bs, num_queries, _ = output.size()
        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)  # bs, num_queries, 4

        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                        reference_points[:, :, None]
                        * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                **kwargs,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.look_forward_twice:
                    intermediate_reference_points.append(new_reference_points)
                else:
                    intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), decoder_intermediate_out_list

        return output, reference_points, decoder_intermediate_out_list


class KSDINOTransformer(DINOTransformer):

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            # MultiScaleDeformableAttention
            if isinstance(m, (MultiScaleDeformableAttention, KSBaseMultiScaleDeformableAttention)
                          ):
                m.init_weights()
        nn.init.normal_(self.level_embeds)

    def forward(
            self,
            multi_level_feats,
            multi_level_masks,
            multi_level_pos_embeds,
            query_embed,
            attn_masks,
            **kwargs,
    ):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        # =============================
        if kwargs.get('ksgt', None):  kwargs['ksgt'].mask = mask_flatten  # Set the mask
        # =============================

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )

        memory, *encoder_intermediate_output_list = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,  # bs, num_token, num_level, 2
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )
        # -------------
        encoder_intermediate_output_list = extract_optional_output(encoder_intermediate_output_list)
        # -------------

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        # output_memory: bs, num_tokens, c
        # output_proposals: bs, num_tokens, 4. unsigmoided.

        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        )  # unsigmoided.

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

        # extract region proposal boxes
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )  # unsigmoided.
        reference_points = topk_coords_unact.detach().sigmoid()
        if query_embed[1] is not None:
            reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)
        init_reference_out = reference_points

        # extract region features
        target_unact = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )
        if self.learnt_init_query:
            target = self.tgt_embed.weight[None].repeat(bs, 1, 1)
        else:
            target = target_unact.detach()
        if query_embed[0] is not None:
            target = torch.cat([query_embed[0], target], 1)

        # decoder
        inter_states, inter_references, *decoder_intermediate_output_list = self.decoder(
            query=target,  # bs, num_queries, embed_dims
            key=memory,  # bs, num_tokens, embed_dims  # Key is not used in cross attention
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=None,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 4
            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,  # bs, nlvl, 2
            attn_masks=attn_masks,
            **kwargs,
        )
        # ----
        decoder_intermediate_output_list = extract_optional_output(decoder_intermediate_output_list)
        intermediate_output_dict = dict(
            encoder_intermediate_output_list=encoder_intermediate_output_list,
            decoder_intermediate_output_list=decoder_intermediate_output_list,
            # target=target,  # Only for this method
            # pos_embed=pos_embed,  # The shape has changed
            # query_embed=query_embed,  # The shape has changed
        )
        # ------

        inter_references_out = inter_references
        return (
                   inter_states,
                   init_reference_out,
                   inter_references_out,
                   target_unact,
                   topk_coords_unact.sigmoid(),
               ) + (intermediate_output_dict,)



class KSDINODetrMultiAttnTransformer(KSDINOTransformer):

    def _extract_enc_single_attn_head_out(self, memory, mask_flatten, spatial_shapes, query_embed):
        # TODO: need to return reference_points as well, as it is updated. init_reference_out
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        # output_memory: bs, num_tokens, c
        # output_proposals: bs, num_tokens, 4. unsigmoided.

        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        )  # unsigmoided.

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

        # extract region proposal boxes
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )  # unsigmoided.
        reference_points = topk_coords_unact.detach().sigmoid()
        if query_embed[1] is not None:
            reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)
        init_reference_out = reference_points

        # extract region features
        target_unact = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )

        return target_unact, topk_coords_unact

    def append_teacher_encoder_output(self, mask_flatten, spatial_shapes, query_embed,
                                      encoder_intermediate_output_list, target_unact, topk_coords_unact):

        if encoder_intermediate_output_list:
            # Only support the case that the multiple attentions exist in the last encoder layer
            encoder_out = encoder_intermediate_output_list[-1]

            if 'feat_t' in encoder_out:
                # Change the output to a list
                target_unact, topk_coords_unact = [target_unact], [topk_coords_unact]

                teacher_memories = encoder_out['feat_t']

                # Dual attention does not output a list, and we need to transfer it to a list.
                if not isinstance(teacher_memories, (list, tuple)):
                    teacher_memories = [teacher_memories]

                for teacher_memory in teacher_memories:
                    # ----------------------
                    # Important note: Never update init_reference_out
                    target_unact_t, topk_coords_unact_t = self._extract_enc_single_attn_head_out(
                        memory=teacher_memory, mask_flatten=mask_flatten,
                        spatial_shapes=spatial_shapes, query_embed=query_embed
                    )
                    target_unact.append(target_unact_t)
                    topk_coords_unact.append(topk_coords_unact_t)

        return target_unact, topk_coords_unact

    def forward(
            self,
            multi_level_feats,
            multi_level_masks,
            multi_level_pos_embeds,
            query_embed,
            attn_masks,
            **kwargs,
    ):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        # =============================
        if kwargs.get('ksgt', None):  kwargs['ksgt'].mask = mask_flatten  # Set the mask
        # =============================

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )

        memory, *encoder_intermediate_output_list = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,  # bs, num_token, num_level, 2
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )
        # ---------------
        encoder_intermediate_output_list = extract_optional_output(encoder_intermediate_output_list)
        # ---------------

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        # output_memory: bs, num_tokens, c
        # output_proposals: bs, num_tokens, 4. unsigmoided.

        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        )  # unsigmoided.

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

        # extract region proposal boxes
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )  # unsigmoided.
        reference_points = topk_coords_unact.detach().sigmoid()
        if query_embed[1] is not None:
            reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)
        init_reference_out = reference_points

        # extract region features
        target_unact = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )
        # ===================== No other parameters should be modified.
        # Add the teacher output (if any) of the last encoder layer.
        target_unact, topk_coords_unact = self.append_teacher_encoder_output(
            mask_flatten=mask_flatten,
            spatial_shapes=spatial_shapes, query_embed=query_embed,
            encoder_intermediate_output_list=encoder_intermediate_output_list,
            target_unact=target_unact, topk_coords_unact=topk_coords_unact
        )
        # =====================

        if self.learnt_init_query:  # went to this branch
            target = self.tgt_embed.weight[None].repeat(bs, 1, 1)
        else:
            target = target_unact.detach()
        if query_embed[0] is not None:  # went to this branch
            target = torch.cat([query_embed[0], target], 1)

        # decoder
        inter_states, inter_references, *decoder_intermediate_output_list = self.decoder(
            query=target,  # bs, num_queries, embed_dims
            key=memory,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=None,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 4
            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,  # bs, nlvl, 2
            attn_masks=attn_masks,
            **kwargs,
        )
        # ----
        decoder_intermediate_output_list = extract_optional_output(decoder_intermediate_output_list)
        intermediate_output_dict = dict(
            encoder_intermediate_output_list=encoder_intermediate_output_list,
            decoder_intermediate_output_list=decoder_intermediate_output_list,
            # target=target,  # Only for this method
            # pos_embed=pos_embed,  # The shape has changed
            # query_embed=query_embed,  # The shape has changed
        )
        # ------

        # =================
        # the above are directly copied from its super class, without any modification.

        if encoder_intermediate_output_list:
            count = 0
            for encoder_out in encoder_intermediate_output_list:
                if 'feat_t' in encoder_out:
                    teacher_memories = encoder_out['feat_t']

                    # Dual attention does not output a list, and we need to transfer it to a list.
                    if not isinstance(teacher_memories, (list, tuple)):
                        teacher_memories = [teacher_memories]

                    count += 1
                    for teacher_memory in teacher_memories:
                        # ----------------------
                        # decoder
                        inter_states_t, inter_references_t, *decoder_intermediate_output_list_t = self.decoder(
                            query=target,  # bs, num_queries, embed_dims
                            key=teacher_memory,  # bs, num_tokens, embed_dims
                            value=teacher_memory,  # bs, num_tokens, embed_dims
                            query_pos=None,
                            key_padding_mask=mask_flatten,  # bs, num_tokens
                            reference_points=reference_points,  # num_queries, 4
                            spatial_shapes=spatial_shapes,  # nlvl, 2
                            level_start_index=level_start_index,  # nlvl
                            valid_ratios=valid_ratios,  # bs, nlvl, 2
                            attn_masks=attn_masks,
                            **kwargs,
                        )
                        # ---------------------- The following sections require careful attention.
                        inter_states, inter_references = extend_student_teacher_decoder_output(
                            hs=inter_states, references=inter_references,
                            hs_t=inter_states_t, references_t=inter_references_t,
                        )

                        # if isinstance(self.decoder.bbox_embed, nn.ModuleList):  # self.decoder.bbox_embed = None
                        # Always return the intermediate out to keep the layer structure, because
                        # self.class_embed, self.bbox_embed = nn.ModuleList, nn.ModuleList, not a single
                        # MLP.
                        # ksgt.teacher_attn_return_no_intermediate_out is handle in detr, not in
                        # the Transformer.
                        # inter_states, inter_references = extend_student_teacher_decoder_output(
                        #     hs=inter_states, references=inter_references,
                        #     references_t=inter_references_t, hs_t=inter_states_t,
                        # )
                        # else:
                        #     ksgt = kwargs.get('ksgt', None)
                        #     # hidden_state: torch.Size([6, 2, 300, 256]), reference, torch.Size([2, 300, 2])
                        #     inter_states, inter_references = concat_student_teacher_decoder_output(
                        #         hs=inter_states, references=inter_references,
                        #         references_t=inter_references_t, hs_t=inter_states_t,
                        #         teacher_attn_return_no_intermediate_out=True if
                        #         ksgt and ksgt.teacher_attn_return_no_intermediate_out else False
                        #     )
                        if decoder_intermediate_output_list_t:
                            decoder_intermediate_output_list_t = decoder_intermediate_output_list_t[0]
                            intermediate_output_dict[
                                'decoder_intermediate_output_list_t'] = decoder_intermediate_output_list_t

            assert 0 < count, 'MultiAttnTransformer requires teacher memory to work '

        inter_references_out = inter_references

        if isinstance(topk_coords_unact, list):
            topk_coords_unact = [z.sigmoid() for z in topk_coords_unact]
        else:
            assert torch.is_tensor(topk_coords_unact)
            topk_coords_unact = topk_coords_unact.sigmoid()
        return (
                   inter_states,
                   init_reference_out,
                   inter_references_out,
                   target_unact,
                   topk_coords_unact,  # topk_coords_unact.sigmoid()
               ) + (intermediate_output_dict,)


class KSDINODetrMultiAttnTransformerFailedVersion(KSDINOTransformer):

    def _extract_enc_single_attn_head_out(self, memory, mask_flatten, spatial_shapes, query_embed):
        # TODO: need to return reference_points as well, as it is updated. init_reference_out
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        # output_memory: bs, num_tokens, c
        # output_proposals: bs, num_tokens, 4. unsigmoided.

        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        )  # unsigmoided.

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

        # extract region proposal boxes
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )  # unsigmoided.
        reference_points = topk_coords_unact.detach().sigmoid()
        if query_embed[1] is not None:
            reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)
        init_reference_out = reference_points

        # extract region features
        target_unact = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )

        return target_unact, topk_coords_unact

    def append_teacher_encoder_output(self, mask_flatten, spatial_shapes, query_embed,
                                      encoder_intermediate_output_list, target_unact, topk_coords_unact):

        if encoder_intermediate_output_list:
            # Only support the case that the multiple attentions exist in the last encoder layer
            encoder_out = encoder_intermediate_output_list[-1]

            if 'feat_t' in encoder_out:
                # Change the output to a list
                target_unact, topk_coords_unact = [target_unact], [topk_coords_unact]

                teacher_memories = encoder_out['feat_t']

                # Dual attention does not output a list, and we need to transfer it to a list.
                if not isinstance(teacher_memories, (list, tuple)):
                    teacher_memories = [teacher_memories]

                for teacher_memory in teacher_memories:
                    # ----------------------
                    # Important note: Never update init_reference_out
                    target_unact_t, topk_coords_unact_t = self._extract_enc_single_attn_head_out(
                        memory=teacher_memory, mask_flatten=mask_flatten,
                        spatial_shapes=spatial_shapes, query_embed=query_embed
                    )
                    target_unact.append(target_unact_t)
                    topk_coords_unact.append(topk_coords_unact_t)

        return target_unact, topk_coords_unact

    def forward(
            self,
            multi_level_feats,
            multi_level_masks,
            multi_level_pos_embeds,
            query_embed,
            attn_masks,
            **kwargs,
    ):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        # =============================
        if kwargs.get('ksgt', None):  kwargs['ksgt'].mask = mask_flatten  # Set the mask
        # =============================

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )

        memory, *encoder_intermediate_output_list = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,  # bs, num_token, num_level, 2
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )
        # ---------------
        encoder_intermediate_output_list = extract_optional_output(encoder_intermediate_output_list)
        # ---------------

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        # output_memory: bs, num_tokens, c
        # output_proposals: bs, num_tokens, 4. unsigmoided.

        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        )  # unsigmoided.

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

        # extract region proposal boxes
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )  # unsigmoided.
        reference_points = topk_coords_unact.detach().sigmoid()
        if query_embed[1] is not None:
            reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)
        init_reference_out = reference_points

        # extract region features
        target_unact = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )
        # ===================== No other parameters should be modified.
        # Add the teacher output (if any) of the last encoder layer.
        target_unact, topk_coords_unact = self.append_teacher_encoder_output(
            mask_flatten=mask_flatten,
            spatial_shapes=spatial_shapes, query_embed=query_embed,
            encoder_intermediate_output_list=encoder_intermediate_output_list,
            target_unact=target_unact, topk_coords_unact=topk_coords_unact
        )
        # =====================

        if self.learnt_init_query:
            target = self.tgt_embed.weight[None].repeat(bs, 1, 1)
        else:
            target = target_unact.detach()
        if query_embed[0] is not None:
            target = torch.cat([query_embed[0], target], 1)

        # decoder
        inter_states, inter_references, *decoder_intermediate_output_list = self.decoder(
            query=target,  # bs, num_queries, embed_dims
            key=memory,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=None,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 4
            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,  # bs, nlvl, 2
            attn_masks=attn_masks,
            **kwargs,
        )
        # ----
        decoder_intermediate_output_list = extract_optional_output(decoder_intermediate_output_list)
        intermediate_output_dict = dict(
            encoder_intermediate_output_list=encoder_intermediate_output_list,
            decoder_intermediate_output_list=decoder_intermediate_output_list,
            # target=target,  # Only for this method
            # pos_embed=pos_embed,  # The shape has changed
            # query_embed=query_embed,  # The shape has changed
        )
        # ------

        # =================
        # the above are directly copied from its super class, without any modification.

        if encoder_intermediate_output_list:
            count = 0
            for encoder_out in encoder_intermediate_output_list:
                if 'feat_t' in encoder_out:
                    teacher_memories = encoder_out['feat_t']

                    # Dual attention does not output a list, and we need to transfer it to a list.
                    if not isinstance(teacher_memories, (list, tuple)):
                        teacher_memories = [teacher_memories]

                    count += 1
                    for teacher_memory in teacher_memories:
                        # ----------------------
                        # decoder
                        inter_states_t, inter_references_t, *decoder_intermediate_output_list_t = self.decoder(
                            query=target,  # bs, num_queries, embed_dims
                            key=teacher_memory,  # bs, num_tokens, embed_dims
                            value=teacher_memory,  # bs, num_tokens, embed_dims
                            query_pos=None,
                            key_padding_mask=mask_flatten,  # bs, num_tokens
                            reference_points=reference_points,  # num_queries, 4
                            spatial_shapes=spatial_shapes,  # nlvl, 2
                            level_start_index=level_start_index,  # nlvl
                            valid_ratios=valid_ratios,  # bs, nlvl, 2
                            attn_masks=attn_masks,
                            **kwargs,
                        )
                        # ---------------------- The following sections require careful attention.
                        inter_states, inter_references = extend_student_teacher_decoder_output(
                            hs=inter_states, references=inter_references,
                            hs_t=inter_states_t, references_t=inter_references_t,
                        )

                        # if isinstance(self.decoder.bbox_embed, nn.ModuleList):  # self.decoder.bbox_embed = None
                        # Always return the intermediate out to keep the layer structure, because
                        # self.class_embed, self.bbox_embed = nn.ModuleList, nn.ModuleList, not a single
                        # MLP.
                        # ksgt.teacher_attn_return_no_intermediate_out is handle in detr, not in
                        # the Transformer.
                        # inter_states, inter_references = extend_student_teacher_decoder_output(
                        #     hs=inter_states, references=inter_references,
                        #     references_t=inter_references_t, hs_t=inter_states_t,
                        # )
                        # else:
                        #     ksgt = kwargs.get('ksgt', None)
                        #     # hidden_state: torch.Size([6, 2, 300, 256]), reference, torch.Size([2, 300, 2])
                        #     inter_states, inter_references = concat_student_teacher_decoder_output(
                        #         hs=inter_states, references=inter_references,
                        #         references_t=inter_references_t, hs_t=inter_states_t,
                        #         teacher_attn_return_no_intermediate_out=True if
                        #         ksgt and ksgt.teacher_attn_return_no_intermediate_out else False
                        #     )
                        if decoder_intermediate_output_list_t:
                            decoder_intermediate_output_list_t = decoder_intermediate_output_list_t[0]
                            intermediate_output_dict[
                                'decoder_intermediate_output_list_t'] = decoder_intermediate_output_list_t

            assert 0 < count, 'MultiAttnTransformer requires teacher memory to work '

        inter_references_out = inter_references

        if isinstance(topk_coords_unact, list):
            topk_coords_unact = [z.sigmoid() for z in topk_coords_unact]
        else:
            assert torch.is_tensor(topk_coords_unact)
            topk_coords_unact = topk_coords_unact.sigmoid()
        return (
                   inter_states,
                   init_reference_out,
                   inter_references_out,
                   target_unact,
                   topk_coords_unact,  # topk_coords_unact.sigmoid()
               ) + (intermediate_output_dict,)
