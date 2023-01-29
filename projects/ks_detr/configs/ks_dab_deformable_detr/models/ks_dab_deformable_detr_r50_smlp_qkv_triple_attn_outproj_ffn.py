from detectron2.config import LazyCall as L

from projects.ks_detr.modeling import (
    KSDabDeformableDetrTransformerEncoder,
    KSDabDeformableDetrTransformerDecoder,
    KSDabDeformableDetrMultiAttnTransformer
)

from .ks_dab_deformable_detr_r50 import model

from projects.ks_detr.configs.ks_sgdts.triple_head_smlp_qkv_share_outpro_ffn import ksgt_module
model.ksgt = ksgt_module

model.transformer = L(KSDabDeformableDetrMultiAttnTransformer)(
    encoder=L(KSDabDeformableDetrTransformerEncoder)(
        embed_dim=256,
        num_heads=8,
        feedforward_dim=2048,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        # num_layers=6,
        encoder_layer_config='regular_5-DeformableTripleAttnShareAttnVOutProjFFN_1',
        post_norm=False,
        num_feature_levels="${..num_feature_levels}",
    ),
    decoder=L(KSDabDeformableDetrTransformerDecoder)(
        embed_dim=256,
        num_heads=8,
        feedforward_dim=2048,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        # num_layers=6,
        decoder_layer_config='regular_6',
        return_intermediate=True,
        num_feature_levels="${..num_feature_levels}",
    ),
    as_two_stage="${..as_two_stage}",
    num_feature_levels=4,
    two_stage_num_proposals="${..num_queries}",
)

