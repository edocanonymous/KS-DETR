# from detectron2.config import LazyCall as L
# from projects.ks_detr.modeling import (SGDT,)


# model.ksgt = L(SGDT)(
#     embed_dim=256,
#     pad_fg_pixel=0,
#     token_scoring_gt_criterion='significance_value',  # significance_value
#     token_scoring_discard_split_criterion='gt_only_exp-no_bg_token_remove',
#     token_masking='sMLP',
#     token_masking_loc='K',
# )

from .ks_dn_detr_r50_multi_attn import model
model.transformer.encoder.encoder_layer_config = 'regularSW_5-DualAttnShareVOutProjFFNV0_1'

