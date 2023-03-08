from .ks_deformable_detr_r101_50ep import (
    train,
    dataloader,
    optimizer,
    # lr_multiplier,
    model,
)


from detrex.config import get_config
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_24ep
train.max_iter = 180000

from ..utils import get_out_dir_from_file_name
train.output_dir = get_out_dir_from_file_name(__file__, with_parent_dir=True)
dataloader.evaluator.output_dir = train.output_dir