from .ks_dn_deformable_detr_r50_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    # model,
)


from ..utils import get_out_dir_from_file_name

# modify model config

from .models.ks_dn_deformable_detr_r50_smlp_qkv_triple_attn_outproj_ffn_v0 import model


train.output_dir = get_out_dir_from_file_name(__file__, with_parent_dir=True)
dataloader.evaluator.output_dir = train.output_dir

