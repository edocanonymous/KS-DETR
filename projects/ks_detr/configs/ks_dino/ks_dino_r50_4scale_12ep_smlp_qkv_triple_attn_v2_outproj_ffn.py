from .ks_dino_r50_4scale_50ep import (
    train,
    dataloader,
    optimizer,
    # lr_multiplier,
    # model,
)

from .models.ks_dino_detr_r50_smlp_qkv_triple_attn_v2_outproj_ffn import model
model.transformer.encoder.encoder_layer_config = 'regular_5-DeformableTripleAttnShareAttnVOutProjFFNV0_1'

# dump the testing results into output_dir for visualization
from ..utils import get_out_dir_from_file_name
train.output_dir = get_out_dir_from_file_name(__file__, with_parent_dir=True)
dataloader.evaluator.output_dir = train.output_dir


from detrex.config import get_config
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train.max_iter = 90000


