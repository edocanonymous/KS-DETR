from .ks_dn_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    # lr_multiplier,
    model,
)

from ..ks_coco_schedule import ks_lr_multiplier_65ep
lr_multiplier = ks_lr_multiplier_65ep

# max training iterations
train.max_iter = 487500