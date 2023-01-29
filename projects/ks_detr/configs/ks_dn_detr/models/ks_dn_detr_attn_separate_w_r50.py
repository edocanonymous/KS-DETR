from .ks_dn_detr_r50 import model

model.transformer.encoder.encoder_layer_config = 'regularSW_6'
# model.transformer.decoder.decoder_layer_config='regular_6'
