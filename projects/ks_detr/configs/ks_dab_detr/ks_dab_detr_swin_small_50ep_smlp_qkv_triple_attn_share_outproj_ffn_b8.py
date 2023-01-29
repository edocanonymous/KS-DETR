from .ks_dab_detr_swin_small_50ep_smlp_qkv_triple_attn_share_outproj_ffn import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)


dataloader.train.total_batch_size = 8
