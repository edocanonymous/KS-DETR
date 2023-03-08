from .ks_dn_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    # model,
)

from .models.ks_dn_detr_r50_smlp_qkv_triple_attn import model

# dump the testing results into output_dir for visualization
from ..utils import get_out_dir_from_file_name
train.output_dir = get_out_dir_from_file_name(__file__, with_parent_dir=True)
dataloader.evaluator.output_dir = train.output_dir



