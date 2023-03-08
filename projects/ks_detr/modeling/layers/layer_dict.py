from torch import nn as nn

from detrex.layers import BaseTransformerLayer, FFN, ConditionalCrossAttention, MultiheadAttention, \
    MultiScaleDeformableAttention

from projects.group_detr.modeling import GroupConditionalSelfAttention

from .attention import (
    KSBaseMultiheadAttention,
    KSMultiheadDualAttentionShareV,
    KSMultiheadDualAttentionShareA,
    KSMultiheadTripleAttentionShareAV,
    KSConditionalSelfAttention,
    KSConditionalCrossAttention,
    KSConditionalCrossAttentionShareV,
    KSConditionalCrossAttentionShareA,
    KSConditionalCrossAttentionTripleAttn,
)
from .multi_scale_deform_attn import (
    KSBaseMultiScaleDeformableAttention,
    KSMultiScaleTripleAttention,
)

from .transformer import (
    KSBaseTransformerLayer,
)
from projects.ks_detr.modeling.ks_utils import parser_encoder_decoder_layers

EncoderLayerDict = {
    'regular': dict(
        LayerType=KSBaseTransformerLayer,
        SelfAttentionType=KSBaseMultiheadAttention,
    ),
    'DualAttnShareV': dict(LayerType=KSBaseTransformerLayer,
                                       SelfAttentionType=KSMultiheadDualAttentionShareV, ),
    'DualAttnShareA': dict(LayerType=KSBaseTransformerLayer,
                                          SelfAttentionType=KSMultiheadDualAttentionShareA, ),
    'TripleAttnQKVShareAV': dict(LayerType=KSBaseTransformerLayer,
                                               SelfAttentionType=KSMultiheadTripleAttentionShareAV, ),
}

DecoderLayerDict = {
    'regular': dict(LayerType=KSBaseTransformerLayer,
                    SelfAttentionType=KSConditionalSelfAttention,
                    CrossAttentionType=KSConditionalCrossAttention,
                    ),
    'TripleAttnShareAV':
        dict(LayerType=KSBaseTransformerLayer,
             SelfAttentionType=KSConditionalSelfAttention,
             CrossAttentionType=KSConditionalCrossAttentionTripleAttn,
             ),
    'DualAttentionShareV':
        dict(LayerType=KSBaseTransformerLayer,
             SelfAttentionType=KSConditionalSelfAttention,
             CrossAttentionType=KSConditionalCrossAttentionShareV,
             ),
    'DualAttentionShareA':
        dict(LayerType=KSBaseTransformerLayer,
             SelfAttentionType=KSConditionalSelfAttention,
             CrossAttentionType=KSConditionalCrossAttentionShareA,
             ),
    'TripleAttnShareKV':
        dict(LayerType=KSBaseTransformerLayer,
             SelfAttentionType=KSConditionalSelfAttention,
             CrossAttentionType=KSConditionalCrossAttentionTripleAttn,
             ),
}

DeformableEncoderLayerDict = {
    'regular': dict(
        LayerType=KSBaseTransformerLayer,
        SelfAttentionType=KSBaseMultiScaleDeformableAttention,

    ),
    'DeformableTripleAttnShareAV': dict(
        LayerType=KSBaseTransformerLayer,
        SelfAttentionType=KSMultiScaleTripleAttention,
    ),
}

DeformableDecoderLayerDict = {
    'regular': dict(LayerType=BaseTransformerLayer,
                    SelfAttentionType=None,
                    CrossAttentionType=None,
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
        batch_first,
):
    encoder_layer_conf_list = parser_encoder_decoder_layers(encoder_layer_config)

    encoder_layer_list = []
    for l_name, num_l in encoder_layer_conf_list:
        assert l_name in encoder_layer_dict and num_l > 0
        single_encoder_layer_config = encoder_layer_dict[l_name]
        encoder_layer = single_encoder_layer_config['LayerType'](
            attn=single_encoder_layer_config['SelfAttentionType'](
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_drop=attn_dropout,
                batch_first=batch_first,
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
                single_decoder_layer_config['SelfAttentionType'](  # ConditionalSelfAttention
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    batch_first=batch_first,
                ),
                single_decoder_layer_config['CrossAttentionType'](  # ConditionalCrossAttention
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
                MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    batch_first=True,
                ),
                MultiScaleDeformableAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                    num_levels=num_feature_levels,
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
