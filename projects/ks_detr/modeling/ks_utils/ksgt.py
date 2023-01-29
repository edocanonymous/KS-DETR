import torch
from torch import nn

from .gt_mask_generator import TokenScoringGTGenerator
from .ks_components import src_key_padding_mask2valid_token_mask
from .smlp import sMLP


# from projects.ks_detr.modeling.ks_loss import KL, MSE
# from projects.ks_detr.modeling.ksgt.ksgt_deprecated import GTRatioOrSigma, init_proposal_processor, FgBgClassifier


class KSGT(nn.Module):

    def __init__(self,
                 token_scoring_gt_criterion: str = None,
                 pad_fg_pixel: int = 0,

                 embed_dim: int = None,
                 token_scoring_discard_split_criterion: str = None,
                 token_masking: str = None,
                 token_masking_loc: str = None,
                 eval_decoder_layer: int = -1,
                 teacher_attn_return_no_intermediate_out: bool = False,
                 eval_and_save_teacher_result: bool = False,
                 ):
        super(KSGT, self).__init__()

        self.token_scoring_gt_criterion = token_scoring_gt_criterion
        self.token_scoring_gt_generator = TokenScoringGTGenerator(
            token_scoring_gt_criterion=token_scoring_gt_criterion,
            pad_fg_pixel=pad_fg_pixel,
            # proposal_scoring=args.proposal_scoring,
            # proposal_token_scoring_gt_criterion=args.proposal_token_scoring_gt_criterion,
        )

        # the parameters of SGDT.
        self.smlp_module = None
        if token_scoring_discard_split_criterion:
            self.smlp_module = sMLP(
                embed_dim=embed_dim,
                token_scoring_discard_split_criterion=token_scoring_discard_split_criterion,
            )

        self.targets = None
        self.ksgt_target_raw = None
        self.ksgt_targets = None
        self.ksgt_multi_scale_targets = None
        self.feat_map_size = None
        self.src_key_padding_mask = None  # (B, H, W), set in detr
        self.mask = None  # set in transformer

        if token_masking:
            assert token_masking in ['sMLP', 'MarkFg1Bg0', ]

        if token_masking_loc:
            assert token_masking_loc in ['X', 'Q', 'K', 'V', 'QK', 'KV',
                                         'MHA_out', 'MHA_feature', 'FFN_out', 'FFN_feature', ]

        self.token_masking = token_masking
        self.token_masking_loc = token_masking_loc

        # self.token_adaption_visualization = args.token_adaption_visualization
        # self.visualization_out_sub_dir = args.visualization_out_sub_dir
        # self.gt_ratio_or_sigma = None
        # self.encoder_layer_config = args.encoder_layer_config

        # self.marking_encoder_layer_fg1_bg0 = args.marking_encoder_layer_fg1_bg0
        self.eval_decoder_layer = eval_decoder_layer
        self.teacher_attn_return_no_intermediate_out = teacher_attn_return_no_intermediate_out

        self.eval_and_save_teacher_result = eval_and_save_teacher_result

    def set_ksgt_targets(self, targets, feat_map_size, padded_img_size):
        # feat_map_size =(h, w)
        self.feat_map_size = feat_map_size
        self.targets = targets

        if self.token_scoring_gt_criterion:
            self.ksgt_target_raw = self.token_scoring_gt_generator.get_gt_raw(
                targets=targets, padded_img_size=padded_img_size)

            # multi_scale_targets
            if isinstance(feat_map_size, list):
                self.ksgt_multi_scale_targets = [None] * len(feat_map_size)
                for k, fm_size in enumerate(feat_map_size):  # [(100, 134), (50, 67), (25, 34), (13, 17)]
                    self.ksgt_multi_scale_targets[k] = self.token_scoring_gt_generator.resize_sig_value_gt(
                        self.ksgt_target_raw, fm_size)  # (N, B)

                self.flatten_and_cat_multi_scale_target()

            else:
                self.ksgt_targets = self.token_scoring_gt_generator.resize_sig_value_gt(
                    self.ksgt_target_raw, feat_map_size)

    def get_input_img_sizes(self):
        return torch.stack([t['size'] for t in self.targets], dim=0)

    def flatten_and_cat_multi_scale_target(self):
        assert isinstance(self.ksgt_multi_scale_targets, list) and isinstance(self.ksgt_multi_scale_targets[0], dict)
        self.ksgt_targets = {}
        for k, v in self.ksgt_multi_scale_targets[0].items():  # dict
            ksgt_target_flatten = []
            for lvl in range(len(self.ksgt_multi_scale_targets)):
                #  self.ksgt_multi_scale_targets[lvl][k]: (N, B), it has been  gt_new.flatten(1).permute(1, 0)
                # in resize_ksgt_target
                ksgt_target_flatten.append(self.ksgt_multi_scale_targets[lvl][k])
            self.ksgt_targets[k] = torch.cat(ksgt_target_flatten, 0)  # (NN, B), NN = N1 + N2 + ...
            # torch.Size([16500, 2])

    @property
    def valid_tokens_float(self):
        # Todo: update this function to get the valid_tokens from mask
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
        # not pass mask inside forward function.
        mask = self.token_mask
        assert x.shape[0] == mask.shape[1] and x.shape[1] == mask.shape[0], \
            f'shape mismatch, expect x (N, B, C), mask (B, N), however, ' \
            f'got x {x.shape}, mask {mask.shape}'
        return self.smlp_module(
            x=x,  # N, B, C  torch.Size([630, 2, 256]),
            mask=mask,  # (B, N) torch.Size([2, 630])
            ksgt_targets=self.ksgt_targets,
            feat_map_size=self.feat_map_size,  # feat_map_size = (h, w)
            # sigma=self.sigma,
            # gt_ratio=self.gt_ratio
            # # reclaim_padded_region=self.reclaim_padded_region
        )
    # -------------------------------
    # def update_sigma(self, cur_step, total_steps):
    #     process = cur_step / total_steps
    #     sigma_multiplier = 1 - process
    #     self.sigma = self.sigma_max * sigma_multiplier
