from .ks_deformable_detr_r101_50ep_smlp_qkv_triple_attn import (
    train,
    dataloader,
    optimizer,
    # lr_multiplier,
    model,
)

from detrex.config import get_config
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train.max_iter = 90000

from ..utils import get_out_dir_from_file_name
train.output_dir = get_out_dir_from_file_name(__file__, with_parent_dir=True)
dataloader.evaluator.output_dir = train.output_dir