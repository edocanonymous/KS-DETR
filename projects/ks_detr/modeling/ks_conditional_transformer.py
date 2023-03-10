import torch
import torch.nn as nn

from detrex.layers import (
    MLP,
    get_sine_pos_embed,
)

# ----------------------
from projects.ks_detr.modeling.layers import KSTransformerLayerSequence
from projects.conditional_detr.modeling import ConditionalDetrTransformer
from .ks_utils import concat_student_teacher_decoder_output, extract_optional_output
from .layers.layer_dict import EncoderLayerDict, DecoderLayerDict, generate_transformer_encoder_layers, \
    generate_transformer_decoder_layers
# ----------------------


class KSConditionalDetrTransformerEncoder(KSTransformerLayerSequence):
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
        super(KSConditionalDetrTransformerEncoder, self).__init__(
            encoder_decoder_layer_list=encoder_layer_list
        )
        # super(KSConditionalDetrTransformerEncoder, self).__init__(
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


class KSConditionalDetrTransformerDecoder(KSTransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.0,
        activation: nn.Module = nn.PReLU(),
        # num_layers: int = None,
        batch_first: bool = False,
        post_norm: bool = True,
        return_intermediate: bool = True,

        decoder_layer_config: str = None,
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
        super(KSConditionalDetrTransformerDecoder, self).__init__(
            encoder_decoder_layer_list=decoder_layer_list
        )
        num_layers = self.num_layers
        # --------------------
        # super(KSConditionalDetrTransformerDecoder, self).__init__(
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
        self.ref_point_head = MLP(self.embed_dim, self.embed_dim, 2, 2)

        self.bbox_embed = None

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
        **kwargs,
    ):
        decoder_intermediate_out_list = [{} for _ in range(len(self.layers))]

        intermediate = []
        reference_points_before_sigmoid = self.ref_point_head(
            query_pos
        )  # [num_queries, batch_size, 2]
        reference_points: torch.Tensor = reference_points_before_sigmoid.sigmoid().transpose(0, 1)

        for idx, layer in enumerate(self.layers):
            obj_center = reference_points[..., :2].transpose(0, 1)  # [num_queries, batch_size, 2]

            # do not apply transform in position in the first decoder layer
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)

            # get sine embedding for the query vector
            query_sine_embed = get_sine_pos_embed(obj_center)
            # apply position transform
            query_sine_embed = query_sine_embed[..., : self.embed_dim] * position_transform

            query: torch.Tensor = layer(
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
            return [
                torch.stack(intermediate).transpose(1, 2),
                reference_points,
            ] + [decoder_intermediate_out_list]

        return query.unsqueeze(0), decoder_intermediate_out_list


class KSConditionalDetrTransformer(ConditionalDetrTransformer):

    def forward(self, x, mask, query_embed, pos_embed,
                **kwargs,  # --
                ):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.view(bs, -1)
        memory, *encoder_intermediate_output_list = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask,

            **kwargs,
        )
        encoder_intermediate_output_list = extract_optional_output(encoder_intermediate_output_list)

        target = torch.zeros_like(query_embed)
        # hidden_state: torch.Size([6, 2, 300, 256]), reference, torch.Size([2, 300, 2])
        hidden_state, references, *decoder_intermediate_output_list = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,

            **kwargs,
        )
        decoder_intermediate_output_list = extract_optional_output(decoder_intermediate_output_list)
        intermediate_output_dict = dict(
            encoder_intermediate_output_list=encoder_intermediate_output_list,
            decoder_intermediate_output_list=decoder_intermediate_output_list,
        )

        if encoder_intermediate_output_list:
            for encoder_out in encoder_intermediate_output_list[-1:]:
                if 'feat_t' in encoder_out:
                    teacher_memories = encoder_out['feat_t']

                    # Dual attention does not output a list, and we need to transfer it to a list.
                    if not isinstance(teacher_memories, (list, tuple)):
                        teacher_memories = [teacher_memories]

                    for teacher_memory in teacher_memories:
                        # ----------------------
                        hs_t, references_t, *decoder_intermediate_output_list_t = self.decoder(
                            query=target,
                            key=teacher_memory,
                            value=teacher_memory,
                            key_pos=pos_embed,
                            query_pos=query_embed,
                            **kwargs,
                        )
                        # ----------------------
                        ksgt = kwargs.get('ksgt', None)
                        # ==============================-
                        # Do not update reference because Conditional Detr does not update its reference inside
                        # its decoder, all its decoder layers share the same reference, and it only records
                        # one reference. If we update it here, the dimension will become from
                        # torch.Size([2, 300, 2]) -> torch.Size([4, 300, 2]), which is not expected.
                        # ==========================-
                        # hidden_state: torch.Size([6, 2, 300, 256]), reference, torch.Size([2, 300, 2])
                        hidden_state, _ = concat_student_teacher_decoder_output(
                            hs=hidden_state, references=references,
                            references_t=references_t, hs_t=hs_t,
                            teacher_attn_return_no_intermediate_out=True
                            if ksgt and ksgt.teacher_attn_return_no_intermediate_out else False
                        )

                        if decoder_intermediate_output_list_t:
                            if 'decoder_intermediate_output_list_t' not in intermediate_output_dict:
                                intermediate_output_dict['decoder_intermediate_output_list_t'] = []
                            decoder_intermediate_output_list_t = decoder_intermediate_output_list_t[0]
                            intermediate_output_dict['decoder_intermediate_output_list_t'].append(
                                decoder_intermediate_output_list_t)

        return hidden_state, references, intermediate_output_dict
