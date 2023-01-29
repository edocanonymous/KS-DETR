from math import sqrt

import einops
import numpy as np
import torch
from einops import rearrange
from torch import nn as nn
from torch.nn import functional as F

from models.ksgt.topk import PerturbedTopKFunction, HardTopK, min_max_norm, \
    extract_patches_from_indicators, extract_patches_from_indices


class TokenFGScoringWGF(nn.Module):
    """ Importance Score (foreground score), Object scale Predictor
    """

    def __init__(self, embed_dim):  # ,  with_global_feature=True
        # embed_dim: channel dim, also is the feature dimension for each token
        super().__init__()
        assert isinstance(embed_dim, int)

        hidden_dim = embed_dim // 2
        self.fg_scoring = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),  # two classes, return logits
            # nn.LogSoftmax(dim=-1)
        )
        # self.with_global_feature = with_global_feature

    def forward(self, x, feat_map_size=None, with_global_feat=False, mask=None):
        """

        Args:
            x: dim: N, B, C, where N is the number of tokens, B is the batch size,
            C is the channel dimension.

        B, C, H, W = x.size()
        local_x = x[:, :C // 2]
        global_x = torch.mean(x[:, C // 2:], keepdim=True, dim=(2, 3))
        x = torch.cat([local_x, global_x.expand(B, C // 2, H, W)], dim=1)
        Returns:

        """
        if with_global_feat:
            N, B, C = x.size()
            local_x = x[:, :, :C // 2]
            if mask is not None:  # B, N
                valid_mask = 1 - mask.permute(1, 0).float()  # bool, B, N -> N, B
                x_new = x * valid_mask.unsqueeze(-1)   # (N, B, C) * (N, B, 1)
                valid_count = torch.sum(valid_mask, keepdim=True, dim=0)  # 1, B
                # (1, B, C // 2)
                global_x = torch.sum(x_new[:, :, C // 2:], keepdim=True, dim=0) / valid_count[:, :, None]
            else:
                # average over all tokens for each feature channel
                global_x = torch.mean(x[:, :, C // 2:], keepdim=True, dim=0)

            x = torch.cat([local_x, global_x.expand(N, B, C // 2)], dim=-1)

        fg_score_logit = self.fg_scoring(x)  # torch.Size([650, 2, 2])
        return fg_score_logit


class TokenScoringWGF(nn.Module):
    """ Importance Score (foreground score), Object scale Predictor
    """

    def __init__(self, embed_dim, num_scale_class=None, with_global_feat=True):
        # embed_dim: channel dim, also is the feature dimension for each token
        super().__init__()
        assert isinstance(embed_dim, int)

        hidden_dim = embed_dim // 2
        self.fg_scoring = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),  # two classes, return logits
            # nn.LogSoftmax(dim=-1)
        )
        # default 2 classes: 0, non-small-scale, 1, small scale.
        self.num_scale_class = num_scale_class if num_scale_class is not None else 2

        self.scale_scoring = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.num_scale_class),  # 3 classes.
            # nn.LogSoftmax(dim=-1)
        )

        self.with_global_feat = with_global_feat

    def forward(self, x, feat_map_size=None, with_global_feat=False, mask=None):
        """

        Args:
            x: dim: N, B, C, where N is the number of tokens, B is the batch size,
            C is the channel dimension.

        B, C, H, W = x.size()
        local_x = x[:, :C // 2]
        global_x = torch.mean(x[:, C // 2:], keepdim=True, dim=(2, 3))
        x = torch.cat([local_x, global_x.expand(B, C // 2, H, W)], dim=1)
        Returns:

        """
        if self.with_global_feat:
            N, B, C = x.size()
            local_x = x[:, :, :C // 2]
            # average over all tokens for each feature channel
            global_x = torch.mean(x[:, :, C // 2:], keepdim=True, dim=0)
            x = torch.cat([local_x, global_x.expand(N, B, C // 2)], dim=-1)

        fg_score_logit = self.fg_scoring(x)  # torch.Size([650, 2, 2])
        scale_score_logit = self.scale_scoring(x)  # torch.Size([650, 2, 3])
        return fg_score_logit, scale_score_logit


class TokenFGSmallScaleScoring(nn.Module):
    """ Importance Score (foreground score), Object scale Predictor
    """

    def __init__(self, embed_dim, num_scale_class=None):  # channel dim, also is the feature dimension for each token
        super().__init__()
        assert isinstance(embed_dim, int)

        hidden_dim = embed_dim // 2
        self.fg_scoring = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),  # two classes, return logits
            # nn.LogSoftmax(dim=-1)
        )
        # default 2 classes: 0, non-small-scale, 1, small scale.
        self.num_scale_class = num_scale_class if num_scale_class is not None else 2

        self.scale_scoring = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.num_scale_class),  # 3 classes.
            # nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        """

        Args:
            x: dim: N, B, C, where N is the number of tokens, B is the batch size,
            C is the channel dimension.

        Returns:

        """
        fg_score_logit = self.fg_scoring(x)  # torch.Size([650, 2, 2])
        scale_score_logit = self.scale_scoring(x)  # torch.Size([650, 2, 3])
        return fg_score_logit, scale_score_logit


class TokenFGScoringSigmoid(nn.Module):  # Significance
    """ Importance Score (foreground score), Object scale Predictor
    """

    def __init__(self, embed_dim):  # channel dim, also is the feature dimension for each token
        super().__init__()
        assert isinstance(embed_dim, int)

        hidden_dim = embed_dim // 2
        self.significance_scoring = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),  # 3 classes.
            # nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, feat_map_size=None, with_global_feat=False, mask=None):
        """ Significant value prediction, < 0.5, bg, > 0.5 fg (smaller object, large significance).
        Args:
            x: dim: N, B, C, where N is the number of tokens, B is the batch size,
            C is the channel dimension.
        Returns:
        """
        significance_score_logit = self.significance_scoring(x)  # torch.Size([650, 2, 3])
        return significance_score_logit


class TokenFGScoringSoftmax(nn.Module):  # Significance
    """ Importance Score (foreground score), Object scale Predictor
    """

    def __init__(self, embed_dim):  # channel dim, also is the feature dimension for each token
        super().__init__()
        assert isinstance(embed_dim, int)

        hidden_dim = embed_dim // 2
        self.significance_scoring = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),  # 3 classes.
            # nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, feat_map_size=None, with_global_feat=False, mask=None):
        """ Significant value prediction, < 0.5, bg, > 0.5 fg (smaller object, large significance).
        Args:
            x: dim: N, B, C, where N is the number of tokens, B is the batch size,
            C is the channel dimension.
        Returns:
        """
        significance_score_logit = self.significance_scoring(x)  # torch.Size([650, 2, 3])
        return significance_score_logit


# =====================
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class TokenScoringWGFDynamicViT(nn.Module):
    """ Importance Score Predictor, with global feature.
    Modified from PredictorLG of dyconvnext.py in DynamicViT.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.in_conv = nn.Sequential(
            LayerNorm(embed_dim, eps=1e-6, data_format="channels_first"),
            # conv2d with kernel_size= 1 is same with linear layer.
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, 2, 1),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input_x, feat_map_size, with_global_feat=False, mask=None):
        """
        =
        Args:
            input_x: N, B, C, where N is the number of tokens, B is the batch size,
            C is the channel dimension.
            mask:
            feat_map_size: (h, w)
        Returns:

        """
        # if self.training and mask is not None:
        #     x1, x2 = input_x
        #     input_x = x1 * mask + x2 * (1 - mask)
        # else:
        #     x1 = input_x
        #     x2 = input_x

        N, B, C = input_x.shape
        h, w = feat_map_size
        # N, B, C = x.size()
        # local_x = x[:, :, :C // 2]
        valid_count = None
        if mask is not None:  # B, N
            valid_mask_float = 1 - mask.permute(1, 0).float()  # bool, B, N -> N, B
            input_x = input_x * valid_mask_float.unsqueeze(-1)  # (N, B, C) * (N, B, 1)
            valid_count = torch.sum(valid_mask_float, keepdim=False, dim=0)  # B
            # valid_count = torch.sum(valid_mask, keepdim=True, dim=0)  # 1, B

        input_x = input_x.permute(1, 2, 0).reshape(B, C, h, w)  # N, B, C -> B, C, N -> B, C, h, w

        x = self.in_conv(input_x)
        B, C, H, W = x.size()

        if with_global_feat:
            local_x = x[:, :C // 2]
            # when calculated global mean, we ignore invalid tokens.
            if valid_count is not None:   # B
                #  (B, C // 2, 1, 1)  torch.Size([2, 128, 1, 1]) / (B, 1, 1, 1)
                global_x = torch.sum(x[:, C // 2:], keepdim=True, dim=(2, 3)) / valid_count[:, None, None, None]
            else:
                global_x = torch.mean(x[:, C // 2:], keepdim=True, dim=(2, 3))
            x = torch.cat([local_x, global_x.expand(B, C // 2, H, W)], dim=1)

        pred_score = self.out_conv(x)  # B, C, H, W
        pred_score = pred_score.flatten(2).permute(2, 0, 1)

        # if self.training:
        #     mask = F.gumbel_softmax(pred_score, hard=True, dim=1)[:, 0:1]
        #     return [x1, x2], mask
        # else:
        #     score = pred_score[:,0]
        #     B, H, W = score.shape
        #     N = H * W
        #     num_keep_node = int(N * ratio)
        #     idx = torch.argsort(score.reshape(B, N), dim=1, descending=True)
        #     idx1 = idx[:, :num_keep_node]
        #     idx2 = idx[:, num_keep_node:]
        #     return input_x, [idx1, idx2]
        return pred_score


class TokenScoringPredictorLG(nn.Module):
    """ Importance Score Predictor, with global feature.
    Modified from PredictorLG of dyconvnext.py in DynamicViT.
    """

    def __init__(self, embed_dim, out_channels=1):
        super().__init__()
        self.in_conv = nn.Sequential(
            LayerNorm(embed_dim, eps=1e-6, data_format="channels_first"),
            # conv2d with kernel_size= 1 is same with linear layer.
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, out_channels, 1),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input_x, feat_map_size, with_global_feat=False, mask=None):
        """
        =
        Args:
            input_x: N, B, C, where N is the number of tokens, B is the batch size,
            C is the channel dimension.
            mask:
            feat_map_size: (h, w)
        Returns:

        """
        # if self.training and mask is not None:
        #     x1, x2 = input_x
        #     input_x = x1 * mask + x2 * (1 - mask)
        # else:
        #     x1 = input_x
        #     x2 = input_x

        N, B, C = input_x.shape
        h, w = feat_map_size
        # N, B, C = x.size()
        # local_x = x[:, :, :C // 2]
        valid_count = None
        if mask is not None:  # B, N
            valid_mask_float = 1 - mask.permute(1, 0).float()  # bool, B, N -> N, B
            input_x = input_x * valid_mask_float.unsqueeze(-1)  # (N, B, C) * (N, B, 1)
            valid_count = torch.sum(valid_mask_float, keepdim=False, dim=0)  # B
            # valid_count = torch.sum(valid_mask, keepdim=True, dim=0)  # 1, B

        input_x = input_x.permute(1, 2, 0).reshape(B, C, h, w)  # N, B, C -> B, C, N -> B, C, h, w

        x = self.in_conv(input_x)
        B, C, H, W = x.size()

        if with_global_feat:
            local_x = x[:, :C // 2]
            # when calculated global mean, we ignore invalid tokens.
            if valid_count is not None:  # B
                #  (B, C // 2, 1, 1)  torch.Size([2, 128, 1, 1]) / (B, 1, 1, 1)
                global_x = torch.sum(x[:, C // 2:], keepdim=True, dim=(2, 3)) / valid_count[:, None, None, None]
            else:
                global_x = torch.mean(x[:, C // 2:], keepdim=True, dim=(2, 3))
            x = torch.cat([local_x, global_x.expand(B, C // 2, H, W)], dim=1)

        pred_score = self.out_conv(x)  # B, C, H, W
        pred_score = pred_score.flatten(2).permute(2, 0, 1)

        # if self.training:
        #     mask = F.gumbel_softmax(pred_score, hard=True, dim=1)[:, 0:1]
        #     return [x1, x2], mask
        # else:
        #     score = pred_score[:,0]
        #     B, H, W = score.shape
        #     N = H * W
        #     num_keep_node = int(N * ratio)
        #     idx = torch.argsort(score.reshape(B, N), dim=1, descending=True)
        #     idx1 = idx[:, :num_keep_node]
        #     idx2 = idx[:, num_keep_node:]
        #     return input_x, [idx1, idx2]
        return pred_score


# =====================
# TODO: adapt dynamic vit to this version
# class TokenScoringPredictorLG(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, embed_dim=384):
#         super().__init__()
#         self.in_conv = nn.Sequential(
#             nn.LayerNorm(embed_dim),
#             nn.Linear(embed_dim, embed_dim),
#             nn.GELU()
#         )
#
#         self.out_conv = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.GELU(),
#             nn.Linear(embed_dim // 2, embed_dim // 4),
#             nn.GELU(),
#             nn.Linear(embed_dim // 4, 1)
#         )
#
#     def forward(self, x):
#         x = self.in_conv(x)
#         B, N, C = x.size()
#         local_x = x[:,:, :C//2]
#
#         global_x = torch.mean(x[:,:, C//2:], dim=1, keepdim=True)
#
#         x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
#         return self.out_conv(x)


def ksgt_token2feat(x, feat_map_size, mask=None):
    """
    N, B, C ->  B, C, h, w
    Args:
        x: N, B, C
        feat_map_size: (h, w)
        mask:

    Returns:

    """
    N, B, C = x.shape
    h, w = feat_map_size

    if mask is not None:  # B, N
        valid_mask_float = 1 - mask.permute(1, 0).float()  # bool, B, N -> N, B
        x = x * valid_mask_float.unsqueeze(-1)  # (N, B, C) * (N, B, 1)

    x = x.permute(1, 2, 0).reshape(B, C, h, w)  # N, B, C -> B, C, N -> B, C, h, w

    return x


def ksgt_feat2token(x):
    """
    B, C, h, w -> N, B, C
    Args:
        x:

    Returns:

    """
    x = x.flatten(2).permute(2, 0, 1)
    return x

# from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
# from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer


#         self.conv1 = nn.Conv2d(in_chans, 32, 3, 2, 0, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.act1 = nn.ReLU(inplace=True)
class TokenScoringConv(nn.Module):
    def __init__(self, embed_dim, num_convs=3, conv_kernel_size=3):
        super().__init__()
        hidden_dim = embed_dim

        self.convs = nn.ModuleList()

        padding = (conv_kernel_size - 1) // 2
        for i in range(num_convs):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, hidden_dim, kernel_size=conv_kernel_size, padding=padding),
                    nn.GroupNorm(32, hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.conv_logits = nn.Conv2d(embed_dim, 2,
                                     kernel_size=conv_kernel_size,
                                     padding=padding)
        # # from Mask R-CNN, fcn_mask_head.py
        # self.convs = ModuleList()
        # for i in range(self.num_convs):
        #     in_channels = (
        #         self.in_channels if i == 0 else self.conv_out_channels)
        #     padding = (self.conv_kernel_size - 1) // 2
        #     self.convs.append(
        #         ConvModule(
        #             in_channels,
        #             self.conv_out_channels,
        #             self.conv_kernel_size,
        #             padding=padding,
        #             # conv_cfg=conv_cfg,
        #             # norm_cfg=norm_cfg,
        #             conv_cfg=None,
        #             norm_cfg=None,
        #         )
        #     )
        # self.conv_logits = build_conv_layer(self.predictor_cfg,
        #                                     logits_in_channel, out_channels, 1)

    # initialization refer to dab_deformable_detr

    def forward(self, input_x, feat_map_size, with_global_feat=False, mask=None):
        x = ksgt_token2feat(input_x, feat_map_size, mask=mask)  # torch.Size([2, 256, 30, 32])

        for conv in self.convs:
            x = conv(x)

        out = self.conv_logits(x)
        out = ksgt_feat2token(out)  # torch.Size([960, 2, 1])

        return out



# class TokenSelection(nn.Module):
#     def __init__(self,  num_samples=500):  # score, k, in_channels, stride=None,
#         super(TokenSelection, self).__init__()
#         # self.k = k
#         # self.anchor_size = int(sqrt(k))
#         # self.stride = stride
#         # self.score = score
#         # self.in_channels = in_channels
#         self.num_samples = num_samples
#
#         # if score == 'tpool':
#         #     self.score_network = PredictorLG(embed_dim=2 * in_channels)
#         #
#         # elif score == 'spatch':
#         #     self.score_network = PredictorLG(embed_dim=in_channels)
#         #     self.init = torch.eye(self.k).unsqueeze(0).unsqueeze(-1).cuda()
#
#     def get_indicator(self, scores, k, sigma):
#         indicator = PerturbedTopKFunction.apply(scores, k, self.num_samples, sigma)
#         indicator = einops.rearrange(indicator, "b k d -> b d k")
#         return indicator
#
#     def get_indices(self, scores, k):
#         indices = HardTopK(k, scores)
#         return indices
#
#     def generate_random_indices(self, b, n, k):
#         indices = []
#         for _ in range(b):
#             indice = np.sort(np.random.choice(n, k, replace=False))
#             indices.append(indice)
#         indices = np.vstack(indices)
#         indices = torch.Tensor(indices).long().cuda()
#         return indices
#
#     def generate_uniform_indices(self, b, n, k):
#         indices = torch.linspace(0, n - 1, steps=k).long()
#         indices = indices.unsqueeze(0).cuda()
#         indices = indices.repeat(b, 1)
#         return indices
#
#     def forward(self, x, type, N, T, sigma):
#         # B = x.size(0)
#         # H = W = int(sqrt(N))
#         # indicator = None
#         # indices = None
#         N, B, C = x.shape
#         x_small = torch.zeros_like(x)
#         tokens_small_obj = torch.zeros_like(x) * valid_tokens_float
#
#         if type == 'time':
#             if self.score == 'tpool':
#                 x = rearrange(x, 'b (t n) m -> b t n m', t=T)
#                 avg = torch.mean(x, dim=2, keepdim=False)
#                 max_ = torch.max(x, dim=2).values
#                 x_ = torch.cat((avg, max_), dim=2)
#                 scores = self.score_network(x_).squeeze(-1)
#                 scores = min_max_norm(scores)
#
#                 if self.training:
#                     indicator = self.get_indicator(scores, self.k, sigma)
#                 else:
#                     indices = self.get_indices(scores, self.k)
#                 x = rearrange(x, 'b t n m -> b t (n m)')
#
#         else:
#             s = self.stride if self.stride is not None else int(max((H - self.anchor_size) // 2, 1))
#
#             if self.score == 'spatch':
#                 x = rearrange(x, 'b (t n) c -> (b t) n c', t=T)  # x: b, n, c
#                 scores = self.score_network(x)  # b, n, c (where c = 1)
#                 scores = rearrange(scores, '(b t) (h w) c -> (b t) c h w', b=B, h=H)
#                 scores = F.unfold(scores, kernel_size=self.anchor_size, stride=s)  # B, C, L
#                 scores = scores.mean(dim=1)  # B, L,
#                 scores = min_max_norm(scores)
#
#                 x = rearrange(x, '(b t) (h w) c -> (b t) c h w', b=B, h=H)
#                 x = F.unfold(x, kernel_size=self.anchor_size, stride=s).permute(0, 2, 1).contiguous()
#
#                 if self.training:
#                     indicator = self.get_indicator(scores, 1, sigma)
#
#                 else:
#                     indices = self.get_indices(scores, 1)
#
#         if self.training:
#             if indicator is not None:
#                 patches = extract_patches_from_indicators(x, indicator)
#
#             elif indices is not None:
#                 patches = extract_patches_from_indices(x, indices)
#
#             if type == 'time':
#                 patches = rearrange(patches, 'b k (n c) -> b (k n) c', n=N)
#
#             elif self.score == 'spatch':
#                 patches = rearrange(patches, '(b t) k (c kh kw) -> b (t k kh kw) c',
#                                     b=B, c=self.in_channels, kh=self.anchor_size)
#
#             return patches
#
#
#         else:
#             patches = extract_patches_from_indices(x, indices)
#
#             if type == 'time':
#                 patches = rearrange(patches, 'b k (n c) -> b (k n) c', n=N)
#
#             elif self.score == 'spatch':
#                 patches = rearrange(patches, '(b t) k (c kh kw) -> b (t k kh kw) c',
#                                     b=B, c=self.in_channels, kh=self.anchor_size)
#
#             return patches
#
