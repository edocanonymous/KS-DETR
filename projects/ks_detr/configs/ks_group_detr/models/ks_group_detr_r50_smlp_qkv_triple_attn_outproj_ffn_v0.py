from detectron2.config import LazyCall as L
import torch.nn as nn
from projects.ks_detr.modeling import (
    KSGroupDetrTransformerEncoder,
    KSGroupDetrTransformerDecoder,
    KSGroupDetrMultiAttnTransformer,

)
from .ks_group_detr_r50_g2 import model

from projects.ks_detr.configs.ks_sgdts.triple_head_smlp_qkv_share_outpro_ffn import ksgt_module
model.ksgt = ksgt_module

model.transformer = L(KSGroupDetrMultiAttnTransformer)(
    encoder=L(KSGroupDetrTransformerEncoder)(
        embed_dim=256,
        num_heads=8,
        attn_dropout=0.1,
        feedforward_dim=2048,
        ffn_dropout=0.1,
        activation=L(nn.ReLU)(),
        # num_layers=6,
        encoder_layer_config='regular_5-TripleAttnQKVShareAttnOutProjFFNV0_1',
        post_norm=False,
    ),
    decoder=L(KSGroupDetrTransformerDecoder)(
        embed_dim=256,
        num_heads=8,
        attn_dropout=0.1,
        feedforward_dim=2048,
        ffn_dropout=0.1,
        activation=L(nn.ReLU)(),
        # num_layers=6,
        decoder_layer_config='regular_6',
        group_nums="${...group_nums}",
        post_norm=True,
        return_intermediate=True,
    ),
)
