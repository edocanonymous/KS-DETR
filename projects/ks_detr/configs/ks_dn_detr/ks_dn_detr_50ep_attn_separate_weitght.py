from .ks_dn_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    # model,
)

from .models.ks_dn_detr_attn_separate_w_r50 import model

train.output_dir = "./output/ks_dn_detr_attn_separate_w_r50"


