from .ks_dab_detr_r50_smlp_qkv_triple_attn_outproj_ffn_v0 import model


# DualAttnShareVOutProjFFNV0
# TripleAttnQKVShareAttnOutProjFFNV0_1
model.transformer.encoder.encoder_layer_config = 'regular_5-DualAttnShareVOutProjFFNV0_1'
# DualAttnShareVOutProjFFN DualAttnShareVOutProjFFNV0_1
from projects.ks_detr.configs.ks_sgdts.dual_head_smlp_share_v_outpro_ffn import ksgt_module
model.ksgt = ksgt_module

