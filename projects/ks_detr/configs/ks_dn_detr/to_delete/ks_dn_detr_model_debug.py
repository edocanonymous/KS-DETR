from .ks_dn_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    # model,
)

from detrex.config import get_config

# ---------------------------------
# lr_multiplier_50ep_warmup  # lr_multiplier_50ep
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep

# modify dataloader config
dataloader.train.num_workers = 6  # 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 2

# modify optimizer config
optimizer.lr = 5e-5  # 1e-4

# ---------------------------------

# train.checkpointer.period = 100
train.eval_period = 100
train.output_dir = "./output/debug"


"""
dn_detr_r50.py
ks_dn_detr_attn_separate_w_r50.py
ks_dn_detr_r50.py
ks_dn_detr_r50_share_attn_outproj_ffn.py
ks_dn_detr_r50_share_attn_outproj_ffn_v0.py
ks_dn_detr_r50_share_v_outproj_ffn.py
ks_dn_detr_r50_share_v_outproj_ffn_v0.py
ks_dn_detr_w_gt_r50.py


"""
# from .models.ks_dn_detr_r50_share_attn_outproj_ffn_v0 import model
# from .models.ks_dn_detr_r50_share_attn_outproj_ffn import model
# from .models.ks_dn_detr_r50_share_v_outproj_ffn import model
# from .models.ks_dn_detr_r50_share_v_outproj_ffn_v0 import model
# from .models.ks_dn_detr_r50_smlp_k_share_v_outproj_ffn_v0 import model
# from .models.ks_dn_detr_r50_smlp_qk_share_v_outproj_ffn import model
from .models.ks_dn_detr_w_gt_r50 import model



