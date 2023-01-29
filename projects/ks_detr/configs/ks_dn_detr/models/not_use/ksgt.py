
from projects.ks_detr.modeling import (
    KSGT,
)
from detectron2.config import LazyCall as L
ksgt_module = L(KSGT)(
    embed_dim=256,
    pad_fg_pixel=0,
    token_scoring_gt_criterion='significance_value',  # significance_value
    token_scoring_discard_split_criterion='gt_only_exp-no_bg_token_remove',
    token_masking='sMLP',  # ['sMLP', 'MarkFg1Bg0', ]
    token_masking_loc='QK',  # ['X', 'Q', 'K', 'V', 'QK', 'KV', 'MHA_out', 'MHA_feature', 'FFN_out', 'FFN_feature', ]
    eval_decoder_layer=5,  # Evaluate the last decoder layer of the student branch
    teacher_attn_return_no_intermediate_out=True,
)

