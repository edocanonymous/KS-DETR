import torch
import torch.nn as nn

from detrex.layers import (
    MLP,
    get_sine_pos_embed,
)
from detrex.utils import inverse_sigmoid

from .layers import (
    KSTransformerLayerSequence,
    EncoderLayerDict, DecoderLayerDict,
    generate_transformer_encoder_layers, generate_transformer_decoder_layers
)
from projects.dab_detr.modeling.dab_transformer import DabDetrTransformer
from .ks_utils import concat_student_teacher_decoder_output, extract_optional_output


class KSDabDetrTransformerEncoder(KSTransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.1,
        activation: nn.Module = nn.PReLU(),
        post_norm: bool = False,
        # num_layers: int = 6,
        batch_first: bool = False,

        encoder_layer_config: str = None,  # 'regular_6'
    ):
        # =====================================-
        encoder_layer_list = generate_transformer_encoder_layers(
            encoder_layer_dict=EncoderLayerDict,
            encoder_layer_config=encoder_layer_config,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            feedforward_dim=feedforward_dim,
            ffn_dropout=ffn_dropout,
            activation=activation,
            # post_norm: bool = False,
            batch_first=batch_first,
        )
        super(KSDabDetrTransformerEncoder, self).__init__(
            encoder_decoder_layer_list=encoder_layer_list
        )
        # =====================================-
        # super(KSDabDetrTransformerEncoder, self).__init__(
        #     transformer_layers=BaseTransformerLayer(
        #         attn=MultiheadAttention(
        #             embed_dim=embed_dim,
        #             num_heads=num_heads,
        #             attn_drop=attn_dropout,
        #             batch_first=batch_first,
        #         ),
        #         ffn=FFN(
        #             embed_dim=embed_dim,
        #             feedforward_dim=feedforward_dim,
        #             ffn_drop=ffn_dropout,
        #             activation=activation,
        #         ),
        #         norm=nn.LayerNorm(normalized_shape=embed_dim),
        #         operation_order=("self_attn", "norm", "ffn", "norm"),
        #     ),
        #     num_layers=num_layers,
        # )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)

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
            position_scales = self.query_scale(query)
            query, *intermediate_output = layer(
                query,
                key,
                value,
                query_pos=query_pos * position_scales,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            encoder_intermediate_output_list.append(extract_optional_output(intermediate_output))

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query, encoder_intermediate_output_list


class KSDabDetrTransformerDecoder(KSTransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.0,
        activation: nn.Module = nn.PReLU(),
        # num_layers: int = None,
        modulate_hw_attn: bool = True,
        batch_first: bool = False,
        post_norm: bool = True,
        return_intermediate: bool = True,
        decoder_layer_config: str = None,  # 'regular_6'
    ):
        # --------------------
        decoder_layer_list = generate_transformer_decoder_layers(
            decoder_layer_dict=DecoderLayerDict,
            decoder_layer_config=decoder_layer_config,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            feedforward_dim=feedforward_dim,
            ffn_dropout=ffn_dropout,
            activation=activation,
            # post_norm: bool = False,
            batch_first=batch_first,
        )
        super(KSDabDetrTransformerDecoder, self).__init__(
            encoder_decoder_layer_list=decoder_layer_list
        )
        num_layers = self.num_layers
        # --------------------
        # super(KSDabDetrTransformerDecoder, self).__init__(
        #     transformer_layers=BaseTransformerLayer(
        #         attn=[
        #             ConditionalSelfAttention(
        #                 embed_dim=embed_dim,
        #                 num_heads=num_heads,
        #                 attn_drop=attn_dropout,
        #                 batch_first=batch_first,
        #             ),
        #             ConditionalCrossAttention(
        #                 embed_dim=embed_dim,
        #                 num_heads=num_heads,
        #                 attn_drop=attn_dropout,
        #                 batch_first=batch_first,
        #             ),
        #         ],
        #         ffn=FFN(
        #             embed_dim=embed_dim,
        #             feedforward_dim=feedforward_dim,
        #             ffn_drop=ffn_dropout,
        #             activation=activation,
        #         ),
        #         norm=nn.LayerNorm(
        #             normalized_shape=embed_dim,
        #         ),
        #         operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        #     ),
        #     num_layers=num_layers,
        # )
        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)

        self.bbox_embed = None

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(self.embed_dim, self.embed_dim, 2, 2)
        self.modulate_hw_attn = modulate_hw_attn

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

        for idx in range(num_layers - 1):
            self.layers[idx + 1].attentions[1].query_pos_proj = None

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
        anchor_box_embed=None,
        **kwargs,
    ):

        decoder_intermediate_out_list = [{} for _ in range(len(self.layers))]
        intermediate = []

        reference_boxes = anchor_box_embed.sigmoid()
        intermediate_ref_boxes = [reference_boxes]

        for idx, layer in enumerate(self.layers):
            obj_center = reference_boxes[..., : self.embed_dim]
            query_sine_embed = get_sine_pos_embed(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)

            # do not apply transform in position in the first decoder layer
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)

            # apply position transform
            query_sine_embed = query_sine_embed[..., : self.embed_dim] * position_transform

            if self.modulate_hw_attn:
                ref_hw_cond = self.ref_anchor_head(query).sigmoid()
                query_sine_embed[..., self.embed_dim // 2 :] *= (
                    ref_hw_cond[..., 0] / obj_center[..., 2]
                ).unsqueeze(-1)
                query_sine_embed[..., : self.embed_dim // 2] *= (
                    ref_hw_cond[..., 1] / obj_center[..., 3]
                ).unsqueeze(-1)

            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                is_first_layer=(idx == 0),
                **kwargs,
            )
            if isinstance(query, tuple):
                query, *intermediate_output = query
                intermediate_output = extract_optional_output(intermediate_output)
                decoder_intermediate_out_list[idx].update(intermediate_output)
            else:
                out_dict = {'feat': query}
                decoder_intermediate_out_list[idx].update(out_dict)

            # update anchor boxes after each decoder layer using shared box head.
            if self.bbox_embed is not None:
                # predict offsets and added to the input normalized anchor boxes.
                offsets = self.bbox_embed(query)
                offsets[..., : self.embed_dim] += inverse_sigmoid(reference_boxes)
                new_reference_boxes = offsets[..., : self.embed_dim].sigmoid()

                if idx != self.num_layers - 1:
                    intermediate_ref_boxes.append(new_reference_boxes)
                reference_boxes = new_reference_boxes.detach()

            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(intermediate_ref_boxes).transpose(1, 2),
                ] + [decoder_intermediate_out_list]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_boxes.unsqueeze(0).transpose(1, 2),
                ] + [decoder_intermediate_out_list]

        return query.unsqueeze(0), decoder_intermediate_out_list


class KSDabDetrTransformer(DabDetrTransformer):

    def forward(self, x, mask, anchor_box_embed, pos_embed, **kwargs):
        """
        Added  **kwargs
        Args:
            x:
            mask:
            anchor_box_embed:
            pos_embed:
            **kwargs:

        Returns:

        """
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)  # (c, bs, num_queries)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        anchor_box_embed = anchor_box_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.view(bs, -1)
        memory, *encoder_intermediate_output_list = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask,
            **kwargs,
        )
        if encoder_intermediate_output_list:
            encoder_intermediate_output_list = encoder_intermediate_output_list[0]

        num_queries = anchor_box_embed.shape[0]
        target = torch.zeros(num_queries, bs, self.embed_dim, device=anchor_box_embed.device)

        hidden_state, reference_boxes, *decoder_intermediate_output_list = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            anchor_box_embed=anchor_box_embed,

            **kwargs,
        )

        if decoder_intermediate_output_list:
            decoder_intermediate_output_list = decoder_intermediate_output_list[0]

        intermediate_output_dict = dict(
            encoder_intermediate_output_list=encoder_intermediate_output_list,
            decoder_intermediate_output_list=decoder_intermediate_output_list,
        )

        if encoder_intermediate_output_list:

            for encoder_out in encoder_intermediate_output_list:
                if 'feat_t' in encoder_out:
                    teacher_memories = encoder_out['feat_t']

                    if not isinstance(teacher_memories, (list, tuple)):
                        teacher_memories = [teacher_memories]

                    for teacher_memory in teacher_memories:
                        hs_t, references_t, *decoder_intermediate_output_list_t = self.decoder(
                            query=target,
                            key=teacher_memory,
                            value=teacher_memory,
                            key_pos=pos_embed,
                            anchor_box_embed=anchor_box_embed,
                            **kwargs,
                        )

                        ksgt = kwargs.get('ksgt', None)
                        hidden_state, reference_boxes = concat_student_teacher_decoder_output(
                            hs=hidden_state, references=reference_boxes,
                            references_t=references_t, hs_t=hs_t,
                            teacher_attn_return_no_intermediate_out=True
                            if ksgt and ksgt.teacher_attn_return_no_intermediate_out else False
                        )

                        if decoder_intermediate_output_list_t:
                            decoder_intermediate_output_list_t = decoder_intermediate_output_list_t[0]
                            intermediate_output_dict[
                                'decoder_intermediate_output_list_t'] = decoder_intermediate_output_list_t

        return hidden_state, reference_boxes, intermediate_output_dict
