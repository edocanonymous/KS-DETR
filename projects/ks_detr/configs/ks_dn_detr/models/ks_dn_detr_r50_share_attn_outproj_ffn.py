from .ks_dn_detr_r50_multi_attn import model
model.transformer.encoder.encoder_layer_config = 'regularSW_5-DualAttnShareAttnOutProjFFN_1'

model.transformer = L(KSDNDetrMultiAttnTransformer)(
        encoder=L(KSDNDetrTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=2048,
            ffn_dropout=0.0,
            activation=L(nn.PReLU)(),
            # num_layers=6,
            encoder_layer_config='regularSW_5-DualAttnShareVOutProjFFN_1',
            post_norm=False,
        ),
        decoder=L(KSDNDetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=2048,
            ffn_dropout=0.0,
            activation=L(nn.PReLU)(),
            modulate_hw_attn=True,
            post_norm=True,
            return_intermediate=True,

            # keep this key, as it is referred by several times in decoder initialization and this conf file
            # num_layers=6,
            decoder_layer_config='regular_6',  # encoder_layer_config: 'regular_6',  'regular_4-ksgtv1_1-ksgt_1'
        )
)
