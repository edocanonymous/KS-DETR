from projects.ks_detr.configs.ks_dab_detr.ks_dab_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# --------------------
# modify optimizer config
optimizer.lr = 5e-5   # 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1
# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 4  # 16
# --------------------


# # modify optimizer config
# optimizer.lr = 5e-5   # 1e-4
# # optimizer.betas = (0.9, 0.999)
# # optimizer.weight_decay = 1e-4
# optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1
#
# # modify dataloader config
# dataloader.train.num_workers = 6  # 16
#
# # please notice that this is total batch size.
# # surpose you're using 4 gpus for training and the batch size for
# # each gpu is 16/4 = 4
# dataloader.train.total_batch_size = 4  # 16


