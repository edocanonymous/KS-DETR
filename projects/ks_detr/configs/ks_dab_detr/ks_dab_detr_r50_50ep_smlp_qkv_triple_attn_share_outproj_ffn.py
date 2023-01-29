from .ks_dab_detr_r50_70ep import (
    train,
    dataloader,
    optimizer,
    # lr_multiplier,
    # model,
)

from .models.ks_dab_detr_r50_smlp_qkv_triple_attn_outproj_ffn_v0 import model

# dump the testing results into output_dir for visualization
from ..utils import get_out_dir_from_file_name
train.output_dir = get_out_dir_from_file_name(__file__, with_parent_dir=True)
dataloader.evaluator.output_dir = train.output_dir


from detrex.config import get_config
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
train.max_iter = 375000

