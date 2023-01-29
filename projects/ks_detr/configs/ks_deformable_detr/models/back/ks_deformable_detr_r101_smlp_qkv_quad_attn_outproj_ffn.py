from .ks_deformable_detr_r50_smlp_qkv_triple_attn_outproj_ffn import model

# modify model config
model.backbone.stages.depth = 101

