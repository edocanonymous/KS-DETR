import torch
import torch.nn as nn


class TokenSplit(nn.Module):

    def __init__(self, embed_dim, expand=1):
        super().__init__()
        assert isinstance(embed_dim, int)

        self.expand = expand
        assert self.expand in [1, 2], 'expand not in [1, 2] can not be handled yet.'

        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expand),  # C -> 2C
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: dim: N, B, C, where N is the number of tokens, B is the batch size,
            C is the channel dimension.
        Returns:

        """

        if self.expand == 1:
            z = self.linear(x)
            return z  # torch.split(z, C, dim=-1)
        else:
            C = x.shape[-1]
            z = self.linear(x)
            # z[:,:, :C], z[:,:, C:]
            # Splits the tensor into chunks. Each chunk is a view of the original tensor.
            return torch.split(z, C, dim=-1)


def get_valid_token_mask(x, mask=None):
    """
    Args:
        x: dim: (N, B, C), where N is the number of tokens, B is the batch size,
        mask: (B, N); 0, valid locations, 1, padding locations.
    Returns: bool()
    """
    N, B, C = x.shape
    if mask is None:
        valid_tokens = torch.ones(N, B).to(x.device).bool()
    else:
        valid_tokens = ~(mask.bool().permute(1, 0))  # (B, N) -> (N, B)
    return valid_tokens


class sMLP(nn.Module):
    # 1x1 conv for fg tokens.

    def __init__(self, embed_dim,):
        super().__init__()
        self.token_split_conv = TokenSplit(embed_dim=embed_dim, expand=1)

    def forward(self, x, mask, ksgt_targets):
        fg_gt = ksgt_targets['fg_gt']  # (N, B), e.g.,  torch.Size([756, 2])
        valid_tokens_float = get_valid_token_mask(x, mask).float()  # (N, B)
        gt_fg_token_mask = fg_gt.bool().float() * valid_tokens_float  # (N, B)

        N, B, C = x.shape
        x_fg = torch.zeros_like(x)
        for k in range(B):
            mask = gt_fg_token_mask[:, k].bool()
            x_k = x[:, k, :][mask]   # x: N, B, C
            x_k_new = self.token_split_conv(x_k)
            x_fg[mask, k, :] += x_k_new

        # Keep bg tokens and update fg tokens to x_fg
        x_new = x.clone() * (1 - gt_fg_token_mask).unsqueeze(-1).repeat(1, 1, x.shape[-1]) + x_fg

        output_dict = {'x': x_new, 'gt_fg_token_mask': gt_fg_token_mask, }

        return output_dict
