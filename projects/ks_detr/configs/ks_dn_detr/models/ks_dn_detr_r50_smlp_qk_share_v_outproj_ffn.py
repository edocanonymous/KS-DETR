from detectron2.config import LazyCall as L
import torch.nn as nn
from projects.ks_detr.modeling import (
    KSGT,
    KSDNDetrTransformerEncoder,
    KSDNDetrTransformerDecoder,
    KSDNDetrMultiAttnTransformer
)
from .ks_dn_detr_r50 import model


model.ksgt = L(KSGT)(
    embed_dim=256,
    pad_fg_pixel=0,
    token_scoring_gt_criterion='significance_value',  # significance_value
    token_scoring_discard_split_criterion='gt_only_exp-no_bg_token_remove',
    token_masking='sMLP',  # ['sMLP', 'MarkFg1Bg0', ]
    token_masking_loc='QK',  # ['X', 'Q', 'K', 'V', 'QK', 'KV', 'MHA_out', 'MHA_feature', 'FFN_out', 'FFN_feature', ]
    eval_decoder_layer=5,
)
model.transformer = L(KSDNDetrMultiAttnTransformer)(
        encoder=L(KSDNDetrTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=2048,
            ffn_dropout=0.0,
            activation=L(nn.PReLU)(),
            # num_layers=6,
            encoder_layer_config='regularSW_5-DualAttnShareVOutProjFFN_1',
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

            # keep this key, as it is referred by several times in decoder initialization and this conf file
            # num_layers=6,
            decoder_layer_config='regular_6',  # encoder_layer_config: 'regular_6',  'regular_4-ksgtv1_1-ksgt_1'
        )
)


