from .ks_deformable_detr_r50 import model

# modify model config
model.backbone.stages.depth = 101

