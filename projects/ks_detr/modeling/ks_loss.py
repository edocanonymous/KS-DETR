import torch
from torch import nn
from torch.nn import functional as F


def KL(input, target, valid_tokens_float=None, top_k=None):
    """https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss
    F.softmax(attn_weights, dim=-1)  in self-attention, so dim should be -1
    Args:
        input:
        target:
                torch.Size([616, 2]) N, B
            a = valid_tokens_float[:, 0].bool()
            b = a.sum()
            c = valid_mask[0, a].sum()
            d = 513 * 513
    Returns:

    """
    input = input.float()  # torch.Size([2, 8, 888, 888])
    target = target.float()  # torch.Size([2, 8, 888, 888])
    # loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32),
    #                 F.softmax(target, dim=-1, dtype=torch.float32))
    bsz, num_heads, src_len = input.shape[:3]
    if valid_tokens_float is not None:  # N, B
        valid_mask = torch.bmm(valid_tokens_float.transpose(1, 0).view(bsz, src_len, 1),
                               valid_tokens_float.transpose(1, 0).view(bsz, 1, src_len),
                               )  # B, N, N
        valid_mask = valid_mask.view(bsz, 1, src_len, src_len).expand(-1, num_heads, -1, -1)

        # B, N_Head, N, N  torch.Size([2, 8, 725, 725])
        # valid_mask = valid_tokens_float.transpose(1, 0).view(bsz, 1, src_len, 1).expand(-1, num_heads, -1, src_len)

        if top_k is not None and top_k > 0:
            src_mask = torch.ones_like(valid_mask)
            input_topk_indices = torch.topk(input, k=top_k, dim=-1).indices
            input_mask = torch.zeros_like(valid_mask).scatter_(-1, index=input_topk_indices, src=src_mask)
            target_topk_indices = torch.topk(target, k=top_k, dim=-1).indices
            target_mask = torch.zeros_like(valid_mask).scatter_(-1, index=target_topk_indices, src=src_mask)
            # invalid token may also included in topk
            final_topk_mask = torch.logical_or(input_mask.bool(), target_mask.bool())
            valid_mask = valid_mask * final_topk_mask.float()

        weight = valid_tokens_float.sum() / torch.ones_like(valid_tokens_float).sum()
        loss = (F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32),
                         F.softmax(target, dim=-1, dtype=torch.float32),
                         reduction='none') * valid_mask).sum() / (bsz * num_heads * src_len * weight)
    else:
        raise NotImplementedError
        # # loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32),
        # #                 F.softmax(target, dim=-1, dtype=torch.float32),
        # #                 reduction='mean')
        # loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32),
        #                 F.softmax(target, dim=-1, dtype=torch.float32),
        #                 reduction='sum') / (bsz * num_heads * src_len)

    return loss


def MSE(input, target, valid_tokens_float=None):
    """https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss

    Returns:

    """
    input = input.float()  # N, B, C
    target = target.float()  # N, B, C
    loss_func = nn.MSELoss(reduction='none')

    N, B, C = input.shape
    if valid_tokens_float is not None:  # N, B; torch.Size([713, 2])
        valid_mask = valid_tokens_float.view(N, B, 1).expand(-1, -1, C)
        weight = valid_tokens_float.sum() / torch.ones_like(valid_tokens_float).sum()
        loss = (loss_func(input, target) * valid_mask).sum() / (B * N * weight)
        # tensor(57.0677, device='cuda:0', grad_fn=<DivBackward0>)
    else:
        raise NotImplementedError
    return loss