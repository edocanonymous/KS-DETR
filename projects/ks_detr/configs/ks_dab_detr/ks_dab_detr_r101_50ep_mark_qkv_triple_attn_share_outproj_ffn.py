from .ks_dab_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    # model,
)

from .models.ks_dab_detr_r101_smlp_qkv_triple_attn_outproj_ffn_v0 import model


train.init_checkpoint = "output/weights/R-101.pkl"

from projects.ks_detr.configs.ks_sgdts.triple_head_mark_qkv_share_outpro_ffn import ksgt_module
model.ksgt = ksgt_module


# dump the testing results into output_dir for visualization
from ..utils import get_out_dir_from_file_name
train.output_dir = get_out_dir_from_file_name(__file__, with_parent_dir=True)
dataloader.evaluator.output_dir = train.output_dir



