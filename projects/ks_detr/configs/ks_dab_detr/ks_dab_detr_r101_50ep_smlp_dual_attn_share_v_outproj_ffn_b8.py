from .ks_dab_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    # model,
)

from .models.ks_dab_detr_r50_smlp_dual_attn_share_v_outproj_ffn import model

# need to conver the weights to torchvision format.
# https://github.com/facebookresearch/detectron2/blob/main/tools/convert-torchvision-to-d2.py
train.init_checkpoint = "output/weights/R-101.pkl"
# modify model config
model.backbone.stages.depth = 101

# dump the testing results into output_dir for visualization
from ..utils import get_out_dir_from_file_name
train.output_dir = get_out_dir_from_file_name(__file__, with_parent_dir=True)
dataloader.evaluator.output_dir = train.output_dir



dataloader.train.total_batch_size = 8
