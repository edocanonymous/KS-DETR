from .ks_dab_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    # model,
)

from .models.ks_dab_detr_r101_smlp_qkv_triple_attn_outproj_ffn_v0 import model

# need to conver the weights to torchvision format.
# https://github.com/facebookresearch/detectron2/blob/main/tools/convert-torchvision-to-d2.py
train.init_checkpoint = "output/weights/R-101.pkl"

# Wrong setting (below)
# train.init_checkpoint = "output/dab_detr_r101_50ep.pth"
# "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
# "detectron2://ImageNetPretrained/torchvision/R-101.pkl" does not exist
# https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl

# dump the testing results into output_dir for visualization
from ..utils import get_out_dir_from_file_name
train.output_dir = get_out_dir_from_file_name(__file__, with_parent_dir=True)
dataloader.evaluator.output_dir = train.output_dir



