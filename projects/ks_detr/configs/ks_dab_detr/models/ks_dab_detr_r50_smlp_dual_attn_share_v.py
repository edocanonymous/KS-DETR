from .ks_dab_detr_r50_smlp_qkv_triple_attn import model


# DualAttnShareV
# TripleAttnQKVShareAV_1
model.transformer.encoder.encoder_layer_config = 'regular_5-DualAttnShareV_1'
# DualAttnShareVOutProjFFN DualAttnShareV_1
from projects.ks_detr.configs.ksgt_configs.dual_attn_smlp_share_v_outproj_ffn import ksgt_module
model.ksgt = ksgt_module

