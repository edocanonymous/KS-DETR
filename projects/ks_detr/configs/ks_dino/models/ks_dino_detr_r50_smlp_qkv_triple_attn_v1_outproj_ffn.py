from detectron2.config import LazyCall as L
import torch.nn as nn
from projects.ks_detr.modeling import (
    KSDINOTransformerEncoder,
    KSDINOTransformerDecoder,
    # KSDINODetrMultiAttnTransformer,
KSDINODetrMultiAttnTransformerV1ShareReferencePoint
)

from .ks_dino_r50 import model

from projects.ks_detr.configs.ks_sgdts.triple_head_smlp_qkv_share_outpro_ffn import ksgt_module
model.ksgt = ksgt_module

model.transformer = L(KSDINODetrMultiAttnTransformerV1ShareReferencePoint)(
        encoder=L(KSDINOTransformerEncoder)(
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
        decoder=L(KSDINOTransformerDecoder)(
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
        num_feature_levels=4,
        two_stage_num_proposals="${..num_queries}",
)



