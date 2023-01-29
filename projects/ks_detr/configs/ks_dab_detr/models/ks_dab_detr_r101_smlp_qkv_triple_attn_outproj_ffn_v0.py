from .ks_dab_detr_r50_smlp_qkv_triple_attn_outproj_ffn_v0 import model

# modify model config
model.backbone.stages.depth = 101

