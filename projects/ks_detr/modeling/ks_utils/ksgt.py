import torch
from torch import nn

from .gt_mask_generator import TokenGTMaskGenerator
from .smlp import sMLP


class KSGT(nn.Module):

    def __init__(self,
                 gt_fg_bg_mask_criterion: str = None,
                 pad_fg_pixel: int = 0,

                 with_smlp: bool = False,
                 embed_dim: int = None,

                 encoder_token_masking: str = None,
                 encoder_token_masking_loc: str = None,
                 decoder_token_masking=None,  # ['sMLP', 'MarkFg1Bg0', ]
                 decoder_token_masking_loc=None,  # ['K', 'V', 'KV']

                 eval_decoder_layer: int = -1,
                 teacher_attn_return_no_intermediate_out: bool = False,
                 ):
        super(KSGT, self).__init__()

        self.gt_fg_bg_mask_criterion = gt_fg_bg_mask_criterion

        self.token_scoring_gt_generator = None
        if gt_fg_bg_mask_criterion:
            self.token_scoring_gt_generator = TokenGTMaskGenerator(
                gt_fg_bg_mask_criterion=gt_fg_bg_mask_criterion,
                pad_fg_pixel=pad_fg_pixel,
            )

        self.smlp_module = sMLP(embed_dim=embed_dim,) if with_smlp else None

        self.targets = None
        self.ksgt_target_raw = None
        self.ksgt_targets = None
        self.ksgt_multi_scale_targets = None
        self.feat_map_size = None
        self.src_key_padding_mask = None  # (B, H, W), set in DETR
        self.mask = None  # (B, N) set in transformer

        if encoder_token_masking:
            assert encoder_token_masking in ['sMLP', 'MarkFg1Bg0', ]
        if encoder_token_masking_loc:
            assert encoder_token_masking_loc in ['X', 'Q', 'K', 'V', 'QK', 'KV',
                                         'MHA_out', 'MHA_feature', 'FFN_out', 'FFN_feature', ]
        self.encoder_token_masking = encoder_token_masking
        self.encoder_token_masking_loc = encoder_token_masking_loc

        if decoder_token_masking:
            assert decoder_token_masking in ['sMLP', 'MarkFg1Bg0', ]
        if decoder_token_masking_loc:
            assert decoder_token_masking_loc in ['K', 'V', 'KV', ]
        self.decoder_token_masking = decoder_token_masking
        self.decoder_token_masking_loc = decoder_token_masking_loc

        self.eval_decoder_layer = eval_decoder_layer
        self.teacher_attn_return_no_intermediate_out = teacher_attn_return_no_intermediate_out

    def set_ksgt_targets(self, targets, feat_map_size, padded_img_size):
        # feat_map_size =(h, w)
        self.feat_map_size = feat_map_size
        self.targets = targets

        if self.gt_fg_bg_mask_criterion:
            self.ksgt_target_raw = self.token_scoring_gt_generator.get_gt_raw(
                targets=targets, padded_img_size=padded_img_size)

            # multi_scale_targets
            if isinstance(feat_map_size, list):
                self.ksgt_multi_scale_targets = [None] * len(feat_map_size)
                for k, fm_size in enumerate(feat_map_size):  # [(100, 134), (50, 67), (25, 34), (13, 17)]
                    self.ksgt_multi_scale_targets[k] = self.token_scoring_gt_generator.resize_gt_mask(
                        self.ksgt_target_raw, fm_size)  # (N, B)

                self.flatten_and_cat_multi_scale_target()

            else:
                self.ksgt_targets = self.token_scoring_gt_generator.resize_gt_mask(
                    self.ksgt_target_raw, feat_map_size)

    def get_input_img_sizes(self):
        return torch.stack([t['size'] for t in self.targets], dim=0)

    def flatten_and_cat_multi_scale_target(self):
        assert isinstance(self.ksgt_multi_scale_targets, list) and isinstance(self.ksgt_multi_scale_targets[0], dict)
        self.ksgt_targets = {}
        for k, v in self.ksgt_multi_scale_targets[0].items():  # dict
            ksgt_target_flatten = []
            for lvl in range(len(self.ksgt_multi_scale_targets)):
                # self.ksgt_multi_scale_targets[lvl][k]: (N, B)
                ksgt_target_flatten.append(self.ksgt_multi_scale_targets[lvl][k])
            self.ksgt_targets[k] = torch.cat(ksgt_target_flatten, 0)  # (N_New, B), N_New = N1 + N2 + ...
            # torch.Size([16500, 2])

    @property
    def valid_tokens_float(self):
        if self.mask is not None:
            valid_tokens = ~self.mask.permute(1, 0)  # (B, N) -> (N, B)
        else:
            valid_tokens = src_key_padding_mask2valid_token_mask(self.src_key_padding_mask)
        return valid_tokens.float()

    @property
    def token_mask(self):
        """Return (N,B)"""
        if self.mask is not None:
            return self.mask.bool()
        else:
            mask = self.src_key_padding_mask.flatten(1)  # (B, H, W) -> (B, N)
            return mask.bool()

    def forward(self, x):
        # TODO: give up src_key_padding_mask

        mask = self.token_mask
        assert x.shape[0] == mask.shape[1] and x.shape[1] == mask.shape[0], \
            f'shape mismatch, expect x (N, B, C), mask (B, N), however, ' \
            f'got x {x.shape}, mask {mask.shape}'
        return self.smlp_module(
            x=x,  # N, B, C  torch.Size([630, 2, 256]),
            mask=mask,  # (B, N) torch.Size([2, 630])
            ksgt_targets=self.ksgt_targets,
        )


def src_key_padding_mask2valid_token_mask(src_key_padding_mask):
    assert src_key_padding_mask.dim() == 3  # B, H, W
    valid_tokens = ~(src_key_padding_mask.flatten(1).permute(1, 0))  # (B, N) -> (N, B)
    return valid_tokens
