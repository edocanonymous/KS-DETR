from .ks_dn_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    # model,
)

# from detrex.config import get_config
from .models.ks_dn_detr_w_gt_r50 import model

train.output_dir = "./output/ks_dn_detr_w_gt_r50"


