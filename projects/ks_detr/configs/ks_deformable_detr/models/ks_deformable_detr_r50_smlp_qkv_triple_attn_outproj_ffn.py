from detectron2.config import LazyCall as L
import torch.nn as nn
from projects.ks_detr.modeling import (
    # KSGT, KSDeformableDETR,
    KSDeformableDetrTransformerEncoder,
    KSDeformableDetrTransformerDecoder,
    # KSDeformableDetrTransformer,
    KSDeformableDetrMultiAttnTransformer,
)

from .ks_deformable_detr_r50 import model

from projects.ks_detr.configs.ks_sgdts.triple_head_smlp_qkv_share_outpro_ffn import ksgt_module
model.ksgt = ksgt_module

model.transformer = L(KSDeformableDetrMultiAttnTransformer)(
    encoder=L(KSDeformableDetrTransformerEncoder)(
        embed_dim=256,
        num_heads=8,
        feedforward_dim=1024,
        attn_dropout=0.1,
        ffn_dropout=0.1,
        # num_layers=6,
        encoder_layer_config='regular_5-DeformableTripleAttnShareAttnVOutProjFFN_1',
        post_norm=False,
        num_feature_levels="${..num_feature_levels}",
    ),
    decoder=L(KSDeformableDetrTransformerDecoder)(
        embed_dim=256,
        num_heads=8,
        feedforward_dim=1024,
        attn_dropout=0.1,
        ffn_dropout=0.1,
        decoder_layer_config='regular_6',
        return_intermediate=True,
        num_feature_levels="${..num_feature_levels}",
    ),

    num_feature_levels=4,
    as_two_stage="${..as_two_stage}",
    two_stage_num_proposals="${..num_queries}",
)


