from .smlp_qk_pad0 import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

model.ksgt.pad_fg_pixel = 16

from ...configs.utils import get_out_dir_from_file_name
train.output_dir = get_out_dir_from_file_name(__file__, with_parent_dir=True)
dataloader.evaluator.output_dir = train.output_dir


