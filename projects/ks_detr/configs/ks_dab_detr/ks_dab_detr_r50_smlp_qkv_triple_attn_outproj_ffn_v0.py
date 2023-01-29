from detectron2.config import LazyCall as L
import torch.nn as nn
from projects.ks_detr.modeling import (
    # KSGT, KSDABDETR,
    KSDabDetrMultiAttnTransformer,
    KSDabDetrTransformerEncoder,
    KSDabDetrTransformerDecoder,
    # KSDabDetrTransformer
)
from .ks_dab_detr_r50 import model

# from ...ks_sgdts.triple_head_smlp_qkv_share_outpro_ffn import ksgt_module
from projects.ks_detr.configs.ks_sgdts.triple_head_smlp_qkv_share_outpro_ffn import ksgt_module
model.ksgt = ksgt_module

model.transformer=L(KSDabDetrMultiAttnTransformer)(
    encoder=L(KSDabDetrTransformerEncoder)(
        embed_dim=256,
        num_heads=8,
        attn_dropout=0.0,
        feedforward_dim=2048,
        ffn_dropout=0.0,
        activation=L(nn.PReLU)(),
        # num_layers=6,
        encoder_layer_config='regular_5-TripleAttnQKVShareAttnOutProjFFNV0_1',
    ),
    decoder=L(KSDabDetrTransformerDecoder)(
        embed_dim=256,
        num_heads=8,
        attn_dropout=0.0,
        feedforward_dim=2048,
        ffn_dropout=0.0,
        activation=L(nn.PReLU)(),
        # num_layers=6,
        decoder_layer_config='regular_6',
        modulate_hw_attn=True,
    ),
)

