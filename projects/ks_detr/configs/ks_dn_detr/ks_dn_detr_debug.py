from .ks_dn_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    # model,
)


# modify dataloader config
dataloader.train.num_workers = 6  # 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 2

train.eval_period = 50
# from .models.ks_dn_detr_w_gt_r50 import model
from .models.ks_dn_detr_r50_smlp_qkv_triple_attn_outproj_ffn_v0 import model

