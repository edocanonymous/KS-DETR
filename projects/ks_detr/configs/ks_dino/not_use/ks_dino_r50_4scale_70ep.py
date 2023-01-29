from .ks_dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    # lr_multiplier,
    model,
)

from ..ks_coco_schedule import ks_lr_multiplier_70ep
lr_multiplier = ks_lr_multiplier_70ep
# max training iterations
train.max_iter = 525000

from ..utils import get_out_dir_from_file_name
train.output_dir = get_out_dir_from_file_name(__file__, with_parent_dir=True)
dataloader.evaluator.output_dir = train.output_dir
