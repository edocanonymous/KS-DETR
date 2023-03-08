from detectron2.config import LazyCall as L
import torch.nn as nn
from projects.ks_detr.modeling import (
    KSDabDetrTransformer,
    KSDabDetrTransformerEncoder,
    KSDabDetrTransformerDecoder,
)
from .ks_dab_detr_r50 import model

from projects.ks_detr.configs.ksgt_configs.triple_attn_smlp_share_qkv_outproj_ffn import ksgt_module
model.ksgt = ksgt_module


model.transformer=L(KSDabDetrTransformer)(
    encoder=L(KSDabDetrTransformerEncoder)(
        embed_dim=256,
        num_heads=8,
        attn_dropout=0.0,
        feedforward_dim=2048,
        ffn_dropout=0.0,
        activation=L(nn.PReLU)(),
        # num_layers=6,
        encoder_layer_config='regular_5-TripleAttnQKVShareAV_1',
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

