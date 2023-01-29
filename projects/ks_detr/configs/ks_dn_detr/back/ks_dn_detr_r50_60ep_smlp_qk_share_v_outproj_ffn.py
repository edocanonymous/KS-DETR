from .ks_coco_schedule import ks_lr_multiplier_60ep
from .ks_dn_detr_r50_50ep_smlp_qk_share_v_outproj_ffn import (
    train,
    dataloader,
    optimizer,
    # lr_multiplier,
    model,
)


lr_multiplier = ks_lr_multiplier_60ep

train.output_dir = "./output/ks_dn_detr_r50_smlp_qk_share_v_outproj_ffn"
# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir



