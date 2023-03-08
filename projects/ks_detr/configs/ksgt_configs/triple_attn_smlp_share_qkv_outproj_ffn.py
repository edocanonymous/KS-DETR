from projects.ks_detr.modeling import (
    KSGT,
)
from detectron2.config import LazyCall as L
ksgt_module = L(KSGT)(
    embed_dim=256,
    pad_fg_pixel=0,
    gt_fg_bg_mask_criterion='Fg1Bg0',  # significance_value
    with_smlp=True,
    encoder_token_masking='sMLP',  # ['sMLP', 'MarkFg1Bg0', ]
    encoder_token_masking_loc='X',  # ['X', 'Q', 'K', 'V', 'QK', 'KV', 'MHA_out', 'MHA_feature', 'FFN_out', 'FFN_feature', ]
    eval_decoder_layer=5,  # Which decoder layer to evaluate
    teacher_attn_return_no_intermediate_out=True,
)

