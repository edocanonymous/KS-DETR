from .ks_dab_detr_r50_smlp_qkv_triple_attn import model

model.transformer.encoder.encoder_layer_config = 'regular_5-DualAttnShareA_1'

from projects.ks_detr.configs.ksgt_configs.dual_attn_smlp_share_qk_outproj_ffn import ksgt_module
model.ksgt = ksgt_module
