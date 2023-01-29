# import torch.nn as nn
# from projects.ks_detr.modeling import (
#     # KSDNDETR,
#     # KSDNDetrTransformerEncoder,
#     # KSDNDetrTransformerDecoder,
#     # KSDNDetrTransformer,
#     # KSDNCriterion,
#     KSGT,
# )
from detectron2.config import LazyCall as L
from .ks_dn_detr_r50 import model
from .ksgt import ksgt_module

model.ksgt = ksgt_module
model.transformer.encoder.encoder_layer_config = 'regular_5-AttnWithGT_1'

