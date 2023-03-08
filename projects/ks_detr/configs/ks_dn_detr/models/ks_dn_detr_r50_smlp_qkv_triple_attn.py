from detectron2.config import LazyCall as L
import torch.nn as nn
from projects.ks_detr.modeling import (
    # KSGT,
    KSDNDetrTransformerEncoder,
    KSDNDetrTransformerDecoder,
    KSDNDetrTransformer
)

from .ks_dn_detr_r50 import model

from projects.ks_detr.configs.ksgt_configs.triple_attn_smlp_share_qkv_outproj_ffn import ksgt_module
model.ksgt = ksgt_module

model.transformer = L(KSDNDetrTransformer)(  # The transformer is changed.
        encoder=L(KSDNDetrTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=2048,
            ffn_dropout=0.0,
            activation=L(nn.PReLU)(),
            # num_layers=6,
            encoder_layer_config='regular_5-TripleAttnQKVShareAV_1',
            post_norm=False,
        ),
        decoder=L(KSDNDetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=2048,
            ffn_dropout=0.0,
            activation=L(nn.PReLU)(),
            modulate_hw_attn=True,
            post_norm=True,
            return_intermediate=True,

            decoder_layer_config='regular_6',  # encoder_layer_config: 'regular_6',  'regular_4-ksgtv1_1-ksgt_1'
        )
)



