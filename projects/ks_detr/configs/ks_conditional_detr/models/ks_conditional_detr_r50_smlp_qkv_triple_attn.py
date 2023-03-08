import torch.nn as nn
from detectron2.config import LazyCall as L
from projects.ks_detr.modeling import (
    # KSConditionalDETR,
    # KSConditionalDetrTransformer,
    KSConditionalDetrTransformerDecoder,
    KSConditionalDetrTransformerEncoder,
    KSConditionalDetrTransformer,
    # get_num_of_layer,
)

from .ks_conditional_detr_r50 import model

# from ...ks_sgdts.triple_head_smlp_qkv_share_outpro_ffn import ksgt_module
from projects.ks_detr.configs.ksgt_configs.triple_attn_smlp_share_qkv_outproj_ffn import ksgt_module

model.ksgt = ksgt_module


model.transformer = L(KSConditionalDetrTransformer)(
    encoder=L(KSConditionalDetrTransformerEncoder)(
        embed_dim=256,
        num_heads=8,
        attn_dropout=0.1,
        feedforward_dim=2048,
        ffn_dropout=0.1,
        activation=L(nn.ReLU)(),
        # num_layers=6,
        encoder_layer_config='regular_5-TripleAttnQKVShareAV_1',
        post_norm=False,
    ),
    decoder=L(KSConditionalDetrTransformerDecoder)(
        embed_dim=256,
        num_heads=8,
        attn_dropout=0.1,
        feedforward_dim=2048,
        ffn_dropout=0.1,
        activation=L(nn.ReLU)(),
        # num_layers=6,
        decoder_layer_config='regular_6',
        post_norm=True,
        return_intermediate=True,
    ),
)


