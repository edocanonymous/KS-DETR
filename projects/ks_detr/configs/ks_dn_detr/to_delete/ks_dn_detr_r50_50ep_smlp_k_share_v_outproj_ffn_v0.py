from .ks_dn_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    # model,
)

from .models.ks_dn_detr_r50_smlp_k_share_v_outproj_ffn_v0 import model

train.output_dir = "./output/ks_dn_detr_r50_smlp_k_share_v_outproj_ffn_v0"

# save checkpoint every 5000 iters
train.checkpointer.period = 5000


# modify dataloader config
dataloader.train.num_workers = 10  # 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 16

# modify optimizer config
optimizer.lr = 1e-4  # 5e-5  #
# ---------------------------------

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
