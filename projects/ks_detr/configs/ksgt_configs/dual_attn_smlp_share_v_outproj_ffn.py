from .triple_attn_smlp_share_qkv_outproj_ffn import ksgt_module

# Note, share v means fuse x with GT Fg-Bg Mask to obtain teacher q, k in
# the teacher attention
ksgt_module.encoder_token_masking_loc = 'QK'


