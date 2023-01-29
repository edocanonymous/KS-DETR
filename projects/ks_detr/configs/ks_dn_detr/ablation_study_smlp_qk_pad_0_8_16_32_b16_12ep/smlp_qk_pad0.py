from ..ks_dn_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    # lr_multiplier,
    # model,
)

# from detrex.config import get_config
from ..models.ks_dn_detr_smlp_qk_gt_pad0_r50 import model
from detrex.config import get_config


from ...configs.utils import get_out_dir_from_file_name
train.output_dir = get_out_dir_from_file_name(__file__, with_parent_dir=True)
dataloader.evaluator.output_dir = train.output_dir
# ---------------------------------
# lr_multiplier_50ep_warmup  # lr_multiplier_50ep
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep

# modify dataloader config
dataloader.train.num_workers = 10  # 16

# # please notice that this is total batch size.
# # surpose you're using 4 gpus for training and the batch size for
# # each gpu is 16/4 = 4
# dataloader.train.total_batch_size = 2
