from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import SwinTransformer

from .ks_dab_detr_r50 import model


model.backbone = L(SwinTransformer)(
    embed_dim=96,
    depths=(2, 2, 18, 2),
    num_heads=(3, 6, 12, 24),
    drop_path_rate=0.2,
    out_indices=(3,),
)
# model.backbone = L(SwinTransformer)(
#     pretrain_img_size=224,
#     embed_dim=96,
#     depths=(2, 2, 18, 2),
#     num_heads=(3, 6, 12, 24),
#     drop_path_rate=0.2,
#     window_size=7,
#     out_indices=(1, 2, 3),
# )
model.in_features = ["p3"]
model.in_channels = 768
#
# # modify neck config
# model.neck.input_shapes = {
#     "p1": ShapeSpec(channels=192),
#     "p2": ShapeSpec(channels=384),
#     "p3": ShapeSpec(channels=768),
# }
# model.neck.in_features = ["p1", "p2", "p3"]
