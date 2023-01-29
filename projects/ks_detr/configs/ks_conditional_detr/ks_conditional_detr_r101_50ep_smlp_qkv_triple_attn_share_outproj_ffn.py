from .ks_conditional_detr_r50_50ep_smlp_qkv_triple_attn_share_outproj_ffn import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)


from ..utils import get_out_dir_from_file_name, R101_WEIGHT_PATH

# modify model config
model.backbone.stages.depth = 101
train.init_checkpoint = R101_WEIGHT_PATH

train.output_dir = get_out_dir_from_file_name(__file__, with_parent_dir=True)
dataloader.evaluator.output_dir = train.output_dir
