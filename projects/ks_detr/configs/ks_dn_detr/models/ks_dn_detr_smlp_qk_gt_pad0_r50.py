# import torch.nn as nn
from projects.ks_detr.modeling import (
    # KSDNDETR,
    # KSDNDetrTransformerEncoder,
    # KSDNDetrTransformerDecoder,
    # KSDNDetrTransformer,
    # KSDNCriterion,
    KSGT,
)
from detectron2.config import LazyCall as L
from .ks_dn_detr_r50 import model


model.ksgt = L(KSGT)(
    embed_dim=256,
    pad_fg_pixel=0,
    token_scoring_gt_criterion='significance_value',  # significance_value
    token_scoring_discard_split_criterion='gt_only_exp-no_bg_token_remove',
    token_masking='sMLP',  # ['sMLP', 'MarkFg1Bg0', ]
    token_masking_loc='QK',  # ['X', 'Q', 'K', 'V', 'QK', 'KV', 'MHA_out', 'MHA_feature', 'FFN_out', 'FFN_feature', ]
    eval_decoder_layer=5,  # Evaluate the last decoder layer of the student branch
)

model.transformer.encoder.encoder_layer_config = 'regular_5-AttnWithGT_1'
#
# model.transformer.encoder = L(KSDNDetrTransformerEncoder)(
#     embed_dim=256,
#     num_heads=8,
#     attn_dropout=0.0,
#     feedforward_dim=2048,
#     ffn_dropout=0.0,
#     activation=L(nn.PReLU)(),
#     # num_layers=6,
#     encoder_layer_config='regular_5-AttnWithGT_1',  # encoder_layer_config: 'regular_6',  'regular_4-ksgtv1_1-ksgt_1'
#     post_norm=False,
#
# )


# model.backbone = L(ResNet)(
#     stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
#     stages=L(make_stage)(
#         depth=50,
#         stride_in_1x1=False,
#         norm="FrozenBN",
#         res5_dilation=2,
#     ),
#     out_features=["res2", "res3", "res4", "res5"],
#     freeze_at=1,
# )
