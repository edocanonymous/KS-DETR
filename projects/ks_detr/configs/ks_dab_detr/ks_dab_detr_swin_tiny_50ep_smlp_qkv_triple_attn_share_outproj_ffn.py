from .ks_dab_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    # model,
)


# dump the testing results into output_dir for visualization
from ..utils import get_out_dir_from_file_name
train.output_dir = get_out_dir_from_file_name(__file__, with_parent_dir=True)
dataloader.evaluator.output_dir = train.output_dir


from .models.ks_dab_detr_swin_tiny_smlp_qkv_triple_attn_outproj_ffn_v0 import model

from ..utils import SWIN_TINY_WEIGHT_PATH, SWIN_SMALL_WEIGHT_PATH
train.init_checkpoint = SWIN_TINY_WEIGHT_PATH
#####################
#
# # =================== swin tiny
# from detectron2.config import LazyCall as L
# from detectron2.modeling.backbone import SwinTransformer
# model.backbone = L(SwinTransformer)(
#     embed_dim=96,
#     depths=(2, 2, 6, 2),
#     num_heads=(3, 6, 12, 24),
#     drop_path_rate=0.1,
#     out_indices=(3,),
# )
# model.in_features = ["p3"]
# model.in_channels = 768
#
# from ..utils import SWIN_TINY_WEIGHT_PATH, SWIN_SMALL_WEIGHT_PATH
# train.init_checkpoint = SWIN_TINY_WEIGHT_PATH
# #####################
