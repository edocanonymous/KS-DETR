import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------- from DynamicViT dyconvnext.py without modification
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


class TokenScoring_V0(nn.Module):  # PredictorLG from DynamicViT dyconvnext.py
    """ Importance Score Predictor
    """

    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            LayerNorm(embed_dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, 2, 1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_x, mask=None, ratio=0.5):
        # if self.training and mask is not None:
        #     x1, x2 = input_x
        #     input_x = x1 * mask + x2 * (1 - mask)
        # else:
        #     x1 = input_x
        #     x2 = input_x
        x = self.in_conv(input_x)
        B, C, H, W = x.size()
        local_x = x[:, :C // 2]
        global_x = torch.mean(x[:, C // 2:], keepdim=True, dim=(2, 3))
        x = torch.cat([local_x, global_x.expand(B, C // 2, H, W)], dim=1)
        pred_score = self.out_conv(x)

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

        # self.fg_scoring = nn.Sequential(
        #     nn.Conv2d(embed_dim, embed_dim // 2, 1),
        #     nn.GELU(),
        #     # nn.Conv2d(embed_dim // 2, embed_dim // 4, 1),
        #     # nn.GELU(),
        #     nn.Conv2d(embed_dim // 4, 2, 1),
        #     nn.LogSoftmax(dim=1)
        # )


class TokenScoringV1(nn.Module):
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


class TokenScoring(nn.Module):  # Significance
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

    def forward(self, x):
        """ Significant value prediction, < 0.5, bg, > 0.5 fg (smaller object, large significance).
        Args:
            x: dim: N, B, C, where N is the number of tokens, B is the batch size,
            C is the channel dimension.
        Returns:
        """
        significance_score_logit = self.significance_scoring(x)  # torch.Size([650, 2, 3])
        return significance_score_logit


class TokenSplit(nn.Module):
    """ Importance Score (foreground score), Object scale Predictor
    """

    def __init__(self, embed_dim):  # channel dim, also is the feature dimension for each token
        super().__init__()
        assert isinstance(embed_dim, int)

        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),  # C -> 2C
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """

        Args:
            x: dim: N, B, C, where N is the number of tokens, B is the batch size,
            C is the channel dimension.

        Returns:

        """
        C = x.shape[-1]
        z = self.linear(x)
        # z[:,:, :C], z[:,:, C:]
        # Splits the tensor into chunks. Each chunk is a view of the original tensor.
        return torch.split(z, C, dim=-1)


class TokenMerge(nn.Module):
    """ Importance Score (foreground score), Object scale Predictor
    """

    def __init__(self, embed_dim):  # channel dim, also is the feature dimension for each token
        super().__init__()
        assert isinstance(embed_dim, int)

        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),  # C -> 2C
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """

        Args:
            x: dim: N, B, C, where N is the number of tokens, B is the batch size,
            C is the channel dimension.

        Returns:

        """
        C = x.shape[-1]
        z = self.linear(x)
        # z[:,:, :C], z[:,:, C:]
        # Splits the tensor into chunks. Each chunk is a view of the original tensor.
        return torch.split(z, C, dim=-1)


def get_valid_token_mask(x, mask=None):
    """

    Args:
        x: dim: (N, B, C), where N is the number of tokens, B is the batch size,
        mask: (B, N), 0 valid locations, True padding locations.

    Returns: bool()

    """
    N, B, C = x.shape
    if mask is None:
        valid_tokens = torch.ones(N, B).to(x.device).bool()
    else:
        valid_tokens = ~mask.permute(1, 0)  # (B, N) -> (N, B)
    return valid_tokens


class TokenScoringVersionParser:
    def __init__(self, token_scoring_discard_split_criterion):
        self.token_scoring_discard_split_criterion = token_scoring_discard_split_criterion

    def reclaim_padded_region_(self):
        if self.token_scoring_discard_split_criterion.find('reclaim_padded_region') > -1:
            return True
        else:
            return False

    def ignore_bg_label_(self):
        if self.token_scoring_discard_split_criterion.find('ignore_bg_label') > -1:
            return True
        else:
            return False

    def pred_score_(self):
        """
        Making predictions or not
        if  token_scoring_discard_split_criterion in ['v0_with_gt',
                                 'v0_with_gt_and_reclaim_padded_region',
                                 'v0_with_gt_only_reclaim_padded_region'
                                          ]):
             return True

        Returns:


        """
        if self.token_scoring_discard_split_criterion.find('gt') > -1:
            return False
        else:
            return True


class SGDT_module(nn.Module):
    """
    Token adaption module, input: a set of n tokens
    output: a set of n tokens

    1. token scoring
    2. scoring based token merging, split, removal
    """

    # img_size=224, tokens_type='performer', in_chans=3, , token_dim=64
    def __init__(self, embed_dim, max_split_token_num=10000, token_scoring_discard_split_criterion=None,
                 ):  # no restriction with 10000
        super().__init__()
        self.token_scoring_discard_split_criterion = token_scoring_discard_split_criterion
        self.token_scoring_discard_split_criterion_parser = TokenScoringVersionParser(
            token_scoring_discard_split_criterion)
        if token_scoring_discard_split_criterion is not None and self.token_scoring_discard_split_criterion_parser.pred_score_():
            self.token_scoring = TokenScoring(embed_dim=embed_dim)
        else:
            self.token_scoring = None

        self.token_split_conv = TokenSplit(embed_dim=embed_dim)

        self.max_split_token_num = max_split_token_num

    def discard_split_significance_all_fg_w_priority(
            self, x, mask=None,
            reclaim_padded_region=False,
            ignore_bg_label=False
    ):
        """
        scale guided supervision differentiable, but sumbel_softmax also differentiable
        we need to decouple these two processes to check their effects.
        Using predicted labels as selection results is also a good choice. I need to test this as well.
        Args:
            ignore_bg_label: for testing the idea of only using the padded regions
            reclaim_padded_region:
            x:  N, B, C = x.shape
            mask:

        Returns:

        """
        # (N, B, 1) -> (N, B) e.g., torch.Size([572, 2, 1])
        significance_score_logit = self.token_scoring(x).squeeze(-1)  #
        significance_score = F.sigmoid(significance_score_logit)  # torch.Size([572, 2, 1])

        # token sampling
        fg_mask = significance_score >= 0.5  # 0 bg, 1 fg.
        bg_mask = ~fg_mask
        valid_tokens = get_valid_token_mask(x, mask)
        invalid_tokens = ~valid_tokens

        # background tokens
        tokens_to_discard_original = bg_mask.float() * valid_tokens.float()  # (N, B)
        tokens_to_split_original = fg_mask.float() * valid_tokens.float()

        if reclaim_padded_region:
            if not ignore_bg_label:  # predicted bg for valid tokens + padded region
                # TODO: check if detach is needed
                tokens_to_discard_original = bg_mask  # .detach().clone()

                # update significance_score so that tokens for padded regions has low scores (low priority of being
                # sampled. (later we can change this to high).
                significance_score[~valid_tokens] = float("-inf")  # -1e6  # set the score to be very lower

            else:  # 2) padded region only
                tokens_to_discard_original = ~valid_tokens  # .detach().clone()

        assert not(torch.logical_and(tokens_to_discard_original, tokens_to_split_original).any()), \
            'There should be no overlap for tokens_to_discard_original and tokens_to_split'

        x_new, tokens_to_discard, tokens_to_split = self._extract_adapted_token(
            x=x, tokens_to_discard_original=tokens_to_discard_original,
            tokens_small_obj_original=tokens_to_split_original,
            significance_score=significance_score,
            small_scale_score=significance_score
        )

        src_mask_reclaimed = None
        if reclaim_padded_region:
            src_mask_reclaimed = self.get_attn_mask_for_claimed_padded_regions(
                invalid_tokens=invalid_tokens, tokens_to_discard=tokens_to_discard)

        token_dict = {
            'x': x_new,
            'src_mask_reclaimed': src_mask_reclaimed,
            'invalid_tokens': invalid_tokens,
            'tokens_small_obj': tokens_to_split,
            'tokens_to_discard': tokens_to_discard,

            'fg_mask': fg_mask,
            'fg_obj_score': significance_score,
            'tokens_to_discard_original': tokens_to_discard_original,

            'small_scale_score': significance_score,
            'scale_mask': fg_mask,
            'tokens_small_obj_original': tokens_to_split_original,
        }

        fg_score_logit = None
        scale_score_logit = significance_score_logit
        # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]), probability (sum to 1 for each prediction of one token)
        return token_dict, fg_score_logit, scale_score_logit, valid_tokens

    def _extract_adapted_token(self, x, tokens_to_discard_original,
                               tokens_small_obj_original, significance_score,
                               small_scale_score
                               ):
        """
        the number of tokens to split and remove can be controlled by tokens_to_discard_original and
        tokens_small_obj_original, the priority of sampling can be decided by the significance_score or
        small_scale_score, higher score, higher priority of being sampled.
        Args:
            x:
            tokens_to_discard_original:
            tokens_small_obj_original:
            significance_score:
            small_scale_score:

        Returns:

        """
        # N, B, C = x.shape

        tokens_to_discard = tokens_to_discard_original.clone().detach().bool()
        # foreground and small objects, (N, B)
        tokens_small_obj = tokens_small_obj_original.clone().detach().bool()

        # ======================================
        # calculate the number of tokens to split and remove so that number of split = number of remove
        # ======================================
        token_num = torch.stack([
            torch.sum(tokens_to_discard, dim=0),  # token to discard
            torch.sum(tokens_small_obj, dim=0),  # small object tokens
            # to control the maximum number of tokens to split
            torch.ones_like(torch.sum(tokens_to_discard, dim=0)) * self.max_split_token_num,
        ], dim=0)  # (3, B)
        #  (1, B), first column, the count for the first image; second column, second image.
        min_token_num = torch.min(token_num, dim=0)[0]  # 0, value; 1, indices

        # ======================================
        # set the mask for tokens to split and remove based on the decided number
        # ======================================
        # sample discard tokens if there are more tokens_to_discard than tokens_small_obj
        batch_ids = (token_num[0] - min_token_num > 0).nonzero(as_tuple=True)[0]
        # if (token_num[0] > min_token_num).sum() > 0:
        for k in batch_ids:
            # significance_score[tokens_to_discard] will return a 1d vector and thus does not work
            # Returns the indices that sort a tensor along a given dimension
            # in ascending order (in default) by value.
            sorted_ids = torch.argsort(significance_score[tokens_to_discard[:, k], k], descending=True)  #
            ids = sorted_ids[min_token_num[k]:]
            # torch.nonzero return N,1 tensor
            tokens_to_stop_discard = torch.nonzero(tokens_to_discard[:, k]).squeeze(-1)[ids]
            # change its value from True to False to disable discarding
            tokens_to_discard[tokens_to_stop_discard, k] = False

        # if (token_num[1] > min_token_num).sum() > 0:
        batch_ids = (token_num[1] - min_token_num > 0).nonzero(as_tuple=True)[0]
        for k in batch_ids:  # TODO: put this to a function
            sorted_ids = torch.argsort(small_scale_score[tokens_small_obj[:, k], k], descending=True)  #
            ids = sorted_ids[min_token_num[k]:]

            tokens_to_stop_split = torch.nonzero(tokens_small_obj[:, k]).squeeze(-1)[ids]
            tokens_small_obj[tokens_to_stop_split, k] = False

        # Make sure there is no any token to be both removed and split
        assert not torch.any(torch.logical_and(tokens_small_obj, tokens_to_discard))

        x_new = self._reassemble_tokens(
            x=x, tokens_to_discard_original=tokens_to_discard_original,
            tokens_small_obj_original=tokens_small_obj_original,
            tokens_small_obj=tokens_small_obj, tokens_to_discard=tokens_to_discard)

        return x_new, tokens_to_discard, tokens_small_obj

    def _reassemble_tokens(self, x, tokens_to_discard_original, tokens_small_obj_original,
                           tokens_small_obj, tokens_to_discard):
        """

        Args:
            x:
            tokens_to_discard_original: bool()
            tokens_small_obj_original:  bool()
            tokens_small_obj: bool()
            tokens_to_discard: bool()

        Returns:

        """
        N, B, C = x.shape
        x_small = torch.zeros_like(x)
        for k in range(B):
            # different image has different number of tokens to split or merge, thus
            # batch processing is not possible.
            img_small_obj_ids = tokens_small_obj[:, k]
            img_discard_token_ids = tokens_to_discard[:, k]
            x_k = x[:, k, :][img_small_obj_ids]  # M, C, where M is the number of tokens to split
            tokens_small_obj_new = self.token_split_conv(x_k)
            # # x[img_small_obj_ids, k, :] = tokens_small_obj_new[0]
            # # x[img_discard_token_ids, k, :] = tokens_small_obj_new[1]

            x_small[img_small_obj_ids, k, :] += tokens_small_obj_new[0]  # '+' to make it differentiable
            x_small[img_discard_token_ids, k, :] += tokens_small_obj_new[1]

        # TODO: check if logic_add is not differentiable?
        keep_mask = 1 - (
                tokens_to_discard_original.float() * tokens_to_discard.float() +
                tokens_small_obj_original.float() * tokens_small_obj.float())
        x_new = x.clone() * keep_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) + x_small  # t
        return x_new

    def _reassemble_tokens_v0(self, x, tokens_to_discard_original, tokens_small_obj_original,
                              tokens_small_obj, tokens_to_discard):
        # TODO: further analysis why the Below is also differentiable.
        N, B, C = x.shape

        x_new = x.clone()  # torch.zeros_like(x)
        for k in range(B):
            # different image has different number of tokens to split or merge, thus
            # batch processing is not possible.
            img_small_obj_ids = tokens_small_obj[:, k]
            x_k = x[:, k, :][img_small_obj_ids]  # M, C, where M is the number of tokens to split
            tokens_small_obj_new = self.token_split_conv(x_k)

            img_discard_token_ids = tokens_to_discard[:, k]
            # x[img_small_obj_ids, k, :] = tokens_small_obj_new[0]
            # x[img_discard_token_ids, k, :] = tokens_small_obj_new[1]
            x_new[img_small_obj_ids, k, :] = tokens_small_obj_new[0]
            x_new[img_discard_token_ids, k, :] = tokens_small_obj_new[1]

        return x_new

    @staticmethod
    def get_attn_mask_for_claimed_padded_regions(invalid_tokens, tokens_to_discard):
        """ Note: this version is only for adapted tokens as query, original token as key,
        If this is not the case, then it is wrong.

        Refer to F.multi_head_attention_forward() for generating the attn_mask
        F.functional def multi_head_attention_forward()
        #======================================= my solution
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        #==============================================
        Args:
            invalid_tokens: (N, B), bool
            tokens_to_discard: (N, B), bool

        Returns:

        """
        assert invalid_tokens.shape == tokens_to_discard.shape
        N, B = invalid_tokens.shape
        invalid_tokens_reclaimed = torch.logical_and(invalid_tokens, tokens_to_discard)
        invalid_tokens_remain_invalid = torch.logical_and(invalid_tokens, ~invalid_tokens_reclaimed)
        src_mask_reclaimed = torch.full((N, B), False, device=invalid_tokens.device)
        src_mask_reclaimed[invalid_tokens_remain_invalid] = True
        src_mask_reclaimed = src_mask_reclaimed.permute(1, 0)  # N, B ->  B, N

        # The attention mask is exactly the same as only using src_key_padding_mask, as all queries will
        # estimate attention, even some of them will not be used in the next stage by src_key_padding_mask.
        # The keys marked by src_key_padding_mask will not attended the attention calculation for each target.

        # Must be of shape (L, S)(L,S) or (N\cdot\text{num\_heads}, L, S)(N⋅num_heads,L,S), where NN is the batch size,
        # LL is the target sequence length, and SS is the source sequence length

        # attn_mask = torch.full((N, N), False, device=invalid_tokens.device)  # a tensor with full of False

        # # attn_masks = invalid_tokens.new_tensor(B, N, N)
        # attn_masks = []
        # for k in range(B):
        #     mask = attn_mask.clone()
        #     # column of invalid tokens are marked, indicating the corresponding tokens in original token list
        #     # are invalid, we should not access those locations.
        #     # True + True = 2, be careful.
        #     mask = torch.logical_or(mask, invalid_tokens[:, k].unsqueeze(0))  # .expand(N, -1)
        #     # set some rows to be all 'inf' will cause softmax (attention calculation) to result in 'nan'
        #     #     if attn_mask is not None:  (self attention, F.functional._scaled_dot_product_attention)
        #     #         attn += attn_mask
        #     #     attn = softmax(attn, dim=-1)
        #     # So never do the following things.
        #     # mask = torch.logical_or(mask, invalid_tokens_remain_invalid[:, k].unsqueeze(-1).expand(-1, N))
        #     # mask = torch.logical_or(mask, invalid_tokens[:, k].unsqueeze(-1).expand(-1, N))
        #
        #     attn_masks.append(mask)
        # attn_masks = torch.stack(attn_masks, dim=0)  # (B, N, N)
        # attn_masks = attn_masks[:, None, :, :].expand(-1, self.nhead, -1, -1).reshape(
        #     B * self.nhead, N, N)
        #
        # mask: (B, N), 0 valid locations, True padding locations.
        return src_mask_reclaimed

    def discard_split_with_gt_token_score(self, x, mask=None,
                                          reclaim_padded_region=False,
                                          sgdt_targets=None,
                                          ignore_bg_label=False,
                                          priority_setting=None,
                                          ):
        # ----------------
        # three options, 1) bg in gt, 2) padded region, 3) bg in gt + padded region
        # -----------------
        # scale_gt = fg_gt = torch.where(sgdt_targets['scale_gt'] > 0, 1.0, 0.0)  # ------------
        scale_gt = fg_gt = torch.where(sgdt_targets['scale_gt'] > 0, 1, 0)  # torch.int64
        # fg_score_logit, scale_score_logit = self.token_scoring(x)
        fg_score_logit, scale_score_logit = F.one_hot(fg_gt, num_classes=2).float(), \
                                            F.one_hot(scale_gt, num_classes=2).float()
        # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]); torch.float32

        lsx = nn.LogSoftmax(dim=-1)  # exp(lsx) can get the softmax score, here we use lsx for gumbel_softmax
        fg_score, scale_score = lsx(fg_score_logit), lsx(scale_score_logit)
        # fg_score = lsx(fg_score_logit)
        # scale_score = sgdt_targets['scale_gt']

        # token sampling
        # fg_mask = F.gumbel_softmax(fg_score, hard=True, dim=-1)  # 0 bg, 1 fg.
        fg_mask = fg_score_logit  # 0 bg, 1 fg.
        # foreground score as significance_score
        # torch.randn_like(torch.exp(fg_score[:, :, -1]))  # (N, B) torch.Size([650, 2])
        # significance_score =

        #  0: non-small (medium or large); 1: small.
        #  (old definition 0, small, 1, medium, 2, large is deprecated)
        # scale_mask = F.gumbel_softmax(scale_score, hard=True, dim=-1)  # TODO: tau=1 set temperature
        scale_mask = scale_score_logit  # TODO: tau=1 set temperature
        # small_scale_score = torch.randn_like(torch.exp(scale_score[:, :, -1]))  # TODO: test fg_score * small_score
        significance_score = small_scale_score = sgdt_targets['scale_gt'] + \
                                                 1e-6 * torch.rand_like(sgdt_targets['scale_gt'])

        invalid_tokens = ~(get_valid_token_mask(x, mask))
        # to extract tokens inside image region (exclude padded regions)
        # tokens_to_discard = torch.logical_and(fg_mask[:, :, 0].bool(),
        #                                       valid_tokens)  # (N, B) num to discard: tensor([497., 363.]
        # the invalid tokens are marked as bg in the fg_mask
        # tokens_to_discard_original = fg_mask[:, :, 0] * valid_tokens  # (N, B) num to discard: tensor([497., 363.]
        assert fg_mask[:, :, 0][invalid_tokens].sum() == invalid_tokens.sum(), \
            'All invalid_tokens should be regarded as bg, no any invalid_tokens should be fg.'

        # extract discard-tokens, small-object-tokens
        valid_tokens = get_valid_token_mask(x, mask).float()
        # foreground and small objects, (N, B)
        tokens_small_obj_original = fg_mask[:, :, 1] * scale_mask[:, :, 1] * valid_tokens
        # tokens_large_obj = (fg_mask[:, :, 1] * scale_mask[:, :, 2]).bool()  # foreground + large objects
        # Generate new tokens from the old ones, filling the locations of tokens to discard with split tokens

        if not reclaim_padded_region:
            # to extract tokens inside image region (exclude padded regions
            tokens_to_discard_original = fg_mask[:, :, 0] * valid_tokens  # (N, B) num to discard: tensor([497., 363.]
        else:
            if not ignore_bg_label:  # bg in gt + padded region
                # token scoring, return N, B, Num_Classes (2 for fg, 3 for scale).
                # ----------------------
                tokens_to_discard_original = torch.logical_or(fg_mask[:, :, 0].bool(), invalid_tokens.bool()).float()
                # Modification in this version.

                # update significance_score so that tokens for padded regions has low scores (low priority of being
                # sampled. (later we can change this to high).
                significance_score[invalid_tokens] = float("-inf")  # -1e6  # set the score to be very lower

            else:  # 2) padded region only
                tokens_to_discard_original = invalid_tokens.float()

        # attn_masks = self.generate_attention_mask(
        #     x=x_new, tokens_small_obj=tokens_small_obj, tokens_to_discard=tokens_to_discard)

        x_new, tokens_to_discard, tokens_small_obj = self._extract_adapted_token(
            x=x, tokens_to_discard_original=tokens_to_discard_original,
            tokens_small_obj_original=tokens_small_obj_original,
            significance_score=significance_score,
            small_scale_score=small_scale_score
        )

        src_mask_reclaimed = None
        if reclaim_padded_region:
            src_mask_reclaimed = self.get_attn_mask_for_claimed_padded_regions(
                invalid_tokens=invalid_tokens, tokens_to_discard=tokens_to_discard)

        token_dict = {
            'x': x_new,
            'src_mask_reclaimed': src_mask_reclaimed,
            # 'attn_masks': attn_masks,
            # 'token_num': N,
            # 'map_size': [H, W],
            # 'init_grid_size': [H, W],
            # 'idx_token': idx_token,
            # 'agg_weight': agg_weight,
            'invalid_tokens': invalid_tokens,
            'tokens_small_obj': tokens_small_obj,
            'tokens_to_discard': tokens_to_discard,

            'fg_mask': fg_mask,
            'fg_obj_score': significance_score,
            'tokens_to_discard_original': tokens_to_discard_original,

            'small_scale_score': small_scale_score,
            'scale_mask': scale_mask,
            'tokens_small_obj_original': tokens_small_obj_original,
        }
        # not significance_score, small_scale_score but all predictions, fg_score, torch.exp(scale_score)
        return token_dict, fg_score_logit, scale_score_logit, valid_tokens

    def forward(self, x, mask=None,
                # reclaim_padded_region=False,
                sgdt_targets=None  # only for debugging
                ):
        """

        Args:
            x:
            mask:

        Returns:

        """

        # return self.discard_split(x, mask)

        # return self.discard_split_with_gt_token_score(x, mask, sgdt_targets=sgdt_targets)
        if self.token_scoring_discard_split_criterion == 'v0_with_gt':
            # 'reclaim_padded_region' will control weather to reclaim_padded_region,
            return self.discard_split_with_gt_token_score(
                x, mask, reclaim_padded_region=False,
                sgdt_targets=sgdt_targets,
                ignore_bg_label=False
            )
        elif self.token_scoring_discard_split_criterion == 'v0_with_gt_and_reclaim_padded_region':
            return self.discard_split_with_gt_token_score(
                x, mask, reclaim_padded_region=True,
                sgdt_targets=sgdt_targets,
                ignore_bg_label=False
            )
        elif self.token_scoring_discard_split_criterion == 'v0_with_gt_only_reclaim_padded_region':
            return self.discard_split_with_gt_token_score(
                x, mask, reclaim_padded_region=True,
                sgdt_targets=sgdt_targets,
                ignore_bg_label=True
            )
        elif self.token_scoring_discard_split_criterion == 'v1_selection_differentiable':
            return self.discard_split(x, mask)
        elif self.token_scoring_discard_split_criterion == 'pred_significance_all_fg_w_priority':
            return self.discard_split_significance_all_fg_w_priority(x, mask)
        else:
            raise NotImplementedError


class SGDT_module_deprecated(SGDT_module):

    # def generate_attention_mask(self, x, tokens_small_obj, tokens_to_discard):
    #     """
    #         x_small[img_small_obj_ids, k, :] += tokens_small_obj_new[0]
    #         x_small[img_discard_token_ids, k, :] += tokens_small_obj_new[1]
    #     Args:
    #         x:
    #         tokens_small_obj:
    #         tokens_to_discard:
    #
    #     Returns:
    #
    #     """
    #     N, B, C = x.shape
    #     attn_mask = x.new_tensor(torch.full((N, N), False))  # a tensor with full of False
    #
    #     attn_masks = []  #
    #     # Each image has a different attention mask
    #     for k in range(B):
    #         mask = attn_mask.clone()
    #
    #         img_small_obj_ids = tokens_small_obj[:, k]
    #         img_discard_token_ids = tokens_to_discard[:, k]
    #
    #         # # x[img_small_obj_ids, k, :] = tokens_small_obj_new[0]
    #         # # x[img_discard_token_ids, k, :] = tokens_small_obj_new[1]
    #         for i, j in zip(img_small_obj_ids, img_discard_token_ids):
    #             mask[i, j] = mask[j, i] = True
    #         attn_masks.append(mask)
    #     return attn_masks

    def discard_split_with_gt_token_score_v0(self, x, mask=None, sgdt_targets=None):
        N, B, C = x.shape
        # token scoring, return N, B, Num_Classes (2 for fg, 3 for scale).
        fg_gt, scale_gt = sgdt_targets['fg_gt'], sgdt_targets['scale_gt']

        # fg_score_logit, scale_score_logit = self.token_scoring(x)
        fg_score_logit, scale_score_logit = F.one_hot(fg_gt, num_classes=2).float(), \
                                            F.one_hot(scale_gt, num_classes=2).float()
        # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]); torch.float32

        lsx = nn.LogSoftmax(dim=-1)  # exp(lsx) can get the softmax score, here we use lsx for gumbel_softmax
        fg_score, scale_score = lsx(fg_score_logit), lsx(scale_score_logit)

        # token sampling
        # fg_mask = F.gumbel_softmax(fg_score, hard=True, dim=-1)  # 0 bg, 1 fg.
        fg_mask = fg_score_logit  # 0 bg, 1 fg.
        # foreground score as significance_score
        significance_score = torch.randn_like(torch.exp(fg_score[:, :, -1]))  # (N, B) torch.Size([650, 2])

        #  0: non-small (medium or large); 1: small.
        #  (old definition 0, small, 1, medium, 2, large is deprecated)
        # scale_mask = F.gumbel_softmax(scale_score, hard=True, dim=-1)  # TODO: tau=1 set temperature
        scale_mask = scale_score_logit  # TODO: tau=1 set temperature
        small_scale_score = torch.randn_like(torch.exp(scale_score[:, :, -1]))  # TODO: test fg_score * small_score

        # extract discard-tokens, small-object-tokens
        valid_tokens = get_valid_token_mask(x, mask).float()
        # to extract tokens inside image region (exclude padded regions)
        # tokens_to_discard = torch.logical_and(fg_mask[:, :, 0].bool(),
        #                                       valid_tokens)  # (N, B) num to discard: tensor([497., 363.]
        tokens_to_discard_original = fg_mask[:, :, 0] * valid_tokens  # (N, B) num to discard: tensor([497., 363.]
        # foreground and small objects, (N, B)
        tokens_small_obj_original = fg_mask[:, :, 1] * scale_mask[:, :, 1] * valid_tokens
        # tokens_large_obj = (fg_mask[:, :, 1] * scale_mask[:, :, 2]).bool()  # foreground + large objects
        # Generate new tokens from the old ones, filling the locations of tokens to discard with split tokens

        x_new, tokens_to_discard, tokens_small_obj = self._extract_adapted_token(
            x=x, tokens_to_discard_original=tokens_to_discard_original,
            tokens_small_obj_original=tokens_small_obj_original,
            significance_score=significance_score,
            small_scale_score=small_scale_score
        )
        # attn_masks = self.generate_attention_mask(
        #     x=x_new, tokens_small_obj=tokens_small_obj, tokens_to_discard=tokens_to_discard)

        token_dict = {
            'x': x_new,
            # 'attn_masks': attn_masks,
            # 'token_num': N,
            # 'map_size': [H, W],
            # 'init_grid_size': [H, W],
            # 'idx_token': idx_token,
            # 'agg_weight': agg_weight,
            'tokens_small_obj': tokens_small_obj,
            'tokens_to_discard': tokens_to_discard,

            'fg_mask': fg_mask,
            'fg_obj_score': significance_score,
            'tokens_to_discard_original': tokens_to_discard_original,

            'small_scale_score': small_scale_score,
            'scale_mask': scale_mask,
            'tokens_small_obj_original': tokens_small_obj_original,
        }
        # not significance_score, small_scale_score but all predictions, fg_score, torch.exp(scale_score)
        return token_dict, fg_score_logit, scale_score_logit, valid_tokens
        # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]), probability (sum to 1 for each prediction of one token)

    def discard_split_with_gt_token_score_v1(self, x, mask=None,
                                             reclaim_padded_region=False,
                                             sgdt_targets=None,
                                             ignore_bg_label=False,
                                             priority_setting=None,
                                             ):
        # ----------------
        # three options, 1) bg in gt, 2) padded region, 3) bg in gt + padded region
        # -----------------
        fg_gt, scale_gt = sgdt_targets['fg_gt'], sgdt_targets['scale_gt']

        # fg_score_logit, scale_score_logit = self.token_scoring(x)
        fg_score_logit, scale_score_logit = F.one_hot(fg_gt, num_classes=2).float(), \
                                            F.one_hot(scale_gt, num_classes=2).float()
        # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]); torch.float32

        lsx = nn.LogSoftmax(dim=-1)  # exp(lsx) can get the softmax score, here we use lsx for gumbel_softmax
        fg_score, scale_score = lsx(fg_score_logit), lsx(scale_score_logit)

        # token sampling
        # fg_mask = F.gumbel_softmax(fg_score, hard=True, dim=-1)  # 0 bg, 1 fg.
        fg_mask = fg_score_logit  # 0 bg, 1 fg.
        # foreground score as significance_score
        significance_score = torch.randn_like(torch.exp(fg_score[:, :, -1]))  # (N, B) torch.Size([650, 2])

        #  0: non-small (medium or large); 1: small.
        #  (old definition 0, small, 1, medium, 2, large is deprecated)
        # scale_mask = F.gumbel_softmax(scale_score, hard=True, dim=-1)  # TODO: tau=1 set temperature
        scale_mask = scale_score_logit  # TODO: tau=1 set temperature
        small_scale_score = torch.randn_like(torch.exp(scale_score[:, :, -1]))  # TODO: test fg_score * small_score

        invalid_tokens = ~(get_valid_token_mask(x, mask))
        # to extract tokens inside image region (exclude padded regions)
        # tokens_to_discard = torch.logical_and(fg_mask[:, :, 0].bool(),
        #                                       valid_tokens)  # (N, B) num to discard: tensor([497., 363.]
        # the invalid tokens are marked as bg in the fg_mask
        # tokens_to_discard_original = fg_mask[:, :, 0] * valid_tokens  # (N, B) num to discard: tensor([497., 363.]
        assert fg_mask[:, :, 0][invalid_tokens].sum() == invalid_tokens.sum(), \
            'All invalid_tokens should be regarded as bg, no any invalid_tokens should be fg.'

        # extract discard-tokens, small-object-tokens
        valid_tokens = get_valid_token_mask(x, mask).float()
        # foreground and small objects, (N, B)
        tokens_small_obj_original = fg_mask[:, :, 1] * scale_mask[:, :, 1] * valid_tokens
        # tokens_large_obj = (fg_mask[:, :, 1] * scale_mask[:, :, 2]).bool()  # foreground + large objects
        # Generate new tokens from the old ones, filling the locations of tokens to discard with split tokens

        if not reclaim_padded_region:
            # to extract tokens inside image region (exclude padded regions
            tokens_to_discard_original = fg_mask[:, :, 0] * valid_tokens  # (N, B) num to discard: tensor([497., 363.]
        else:
            if not ignore_bg_label:  # bg in gt + padded region
                # token scoring, return N, B, Num_Classes (2 for fg, 3 for scale).
                # ----------------------
                tokens_to_discard_original = torch.logical_or(fg_mask[:, :, 0].bool(), invalid_tokens.bool()).float()
                # Modification in this version.

                # update significance_score so that tokens for padded regions has low scores (low priority of being
                # sampled. (later we can change this to high).
                significance_score[invalid_tokens] = float("-inf")  # -1e6  # set the score to be very lower

            else:  # 2) padded region only
                tokens_to_discard_original = invalid_tokens.float()

        # attn_masks = self.generate_attention_mask(
        #     x=x_new, tokens_small_obj=tokens_small_obj, tokens_to_discard=tokens_to_discard)

        x_new, tokens_to_discard, tokens_small_obj = self._extract_adapted_token(
            x=x, tokens_to_discard_original=tokens_to_discard_original,
            tokens_small_obj_original=tokens_small_obj_original,
            significance_score=significance_score,
            small_scale_score=small_scale_score
        )

        src_mask_reclaimed = None
        if reclaim_padded_region:
            src_mask_reclaimed = self.get_attn_mask_for_claimed_padded_regions(
                invalid_tokens=invalid_tokens, tokens_to_discard=tokens_to_discard)

        token_dict = {
            'x': x_new,
            'src_mask_reclaimed': src_mask_reclaimed,
            # 'attn_masks': attn_masks,
            # 'token_num': N,
            # 'map_size': [H, W],
            # 'init_grid_size': [H, W],
            # 'idx_token': idx_token,
            # 'agg_weight': agg_weight,
            'invalid_tokens': invalid_tokens,
            'tokens_small_obj': tokens_small_obj,
            'tokens_to_discard': tokens_to_discard,

            'fg_mask': fg_mask,
            'fg_obj_score': significance_score,
            'tokens_to_discard_original': tokens_to_discard_original,

            'small_scale_score': small_scale_score,
            'scale_mask': scale_mask,
            'tokens_small_obj_original': tokens_small_obj_original,
        }
        # not significance_score, small_scale_score but all predictions, fg_score, torch.exp(scale_score)
        return token_dict, fg_score_logit, scale_score_logit, valid_tokens

    def discard_split_v0(self, x, mask=None):
        """
        not differentialable.
                Conduct token adaption inside the module.
                Args:
                    mask: (B, N), 0 valid locations, True padding locations.
                    x: dim: (N, B, C), where N is the number of tokens, B is the batch size,
                    C is the channel dimension.
                    e.g.,
                        x: torch.Size([630, 2, 256]),
                        mask: torch.Size([2, 630])
                Returns:
                """
        N, B, C = x.shape
        # token scoring, return N, B, Num_Classes (2 for fg, 3 for scale).
        fg_score_logit, scale_score_logit = self.token_scoring(
            x)  # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]); torch.float32
        lsx = nn.LogSoftmax(dim=-1)
        fg_score, scale_score = lsx(fg_score_logit), lsx(scale_score_logit)

        # token sampling
        fg_mask = F.gumbel_softmax(fg_score, hard=True, dim=-1)  # 0 bg, 1 fg.
        # foreground score as significance_score
        significance_score = torch.exp(fg_score[:, :, -1])  # (N, B) torch.Size([650, 2])

        #  0: non-small (medium or large); 1: small.
        #  (old definition 0, small, 1, medium, 2, large is deprecated)
        scale_mask = F.gumbel_softmax(scale_score, hard=True, dim=-1)  # TODO: tau=1 set temperature
        small_scale_score = torch.exp(scale_score[:, :, -1])  # TODO: test fg_score * small_score

        # extract discard-tokens, small-object-tokens
        valid_tokens = get_valid_token_mask(x, mask)  # to extract tokens inside image region (exclude padded regions)
        tokens_to_discard = torch.logical_and(fg_mask[:, :, 0].bool(),
                                              valid_tokens)  # (N, B) num to discard: tensor([497., 363.]
        # foreground and small objects, (N, B)
        tokens_small_obj = torch.logical_and((fg_mask[:, :, 1] * scale_mask[:, :, 1]).bool(), valid_tokens)
        # tokens_large_obj = (fg_mask[:, :, 1] * scale_mask[:, :, 2]).bool()  # foreground + large objects

        # calculate the number of tokens to split
        token_num = torch.stack([
            torch.sum(tokens_to_discard, dim=0),  # token to discard
            torch.sum(tokens_small_obj, dim=0),  # small object tokens

            # to control the maximum number of tokens to split
            torch.ones_like(torch.sum(tokens_to_discard, dim=0)) * self.max_split_token_num,
        ], dim=0)  # (3, B)
        #  (1, B), first column, the count for the first image; second column, second image.
        min_token_num = torch.min(token_num, dim=0)[0]  # 0, value; 1, indices

        # sample discard tokens if there are more tokens_to_discard than tokens_small_obj
        batch_ids = (token_num[0] - min_token_num > 0).nonzero(as_tuple=True)[
            0]  # if (token_num[0] > min_token_num).sum() > 0:
        for k in batch_ids:
            # significance_score[tokens_to_discard] will return a 1d vector and thus does not work
            # Returns the indices that sort a tensor along a given dimension
            # in ascending order (in default) by value.
            sorted_ids = torch.argsort(significance_score[tokens_to_discard[:, k], k], descending=True)  #
            ids = sorted_ids[min_token_num[k]:]
            # torch.nonzero return N,1 tensor
            tokens_to_stop_discard = torch.nonzero(tokens_to_discard[:, k]).squeeze(-1)[ids]
            # change its value from True to False to disable discarding
            tokens_to_discard[tokens_to_stop_discard, k] = False

        # if (token_num[1] > min_token_num).sum() > 0:
        batch_ids = (token_num[1] - min_token_num > 0).nonzero(as_tuple=True)[0]
        for k in batch_ids:  # TODO: put this to a function
            sorted_ids = torch.argsort(small_scale_score[tokens_small_obj[:, k], k], descending=True)  #
            ids = sorted_ids[min_token_num[k]:]

            tokens_to_stop_split = torch.nonzero(tokens_small_obj[:, k]).squeeze(-1)[ids]

            tokens_small_obj[tokens_to_stop_split, k] = False

        # Generate new tokens from the old ones, filling the locations of tokens to discard with split tokens
        x_new = x.clone()  # torch.zeros_like(x)
        for k in range(B):
            # different image has different number of tokens to split or merge, thus
            # batch processing is not possible.
            img_small_obj_ids = tokens_small_obj[:, k]
            x_k = x[:, k, :][img_small_obj_ids]  # M, C, where M is the number of tokens to split
            tokens_small_obj_new = self.token_split_conv(x_k)

            img_discard_token_ids = tokens_to_discard[:, k]
            # x[img_small_obj_ids, k, :] = tokens_small_obj_new[0]
            # x[img_discard_token_ids, k, :] = tokens_small_obj_new[1]
            x_new[img_small_obj_ids, k, :] = tokens_small_obj_new[0]
            x_new[img_discard_token_ids, k, :] = tokens_small_obj_new[1]

        # small object tokens
        # token_dict = {'x': x_new,
        #       'token_num': N,
        #       # 'map_size': [H, W],
        #       # 'init_grid_size': [H, W],
        #       'idx_token': idx_token,
        #       # 'agg_weight': agg_weight
        # }
        token_dict = {
            'x': x_new,
            # 'token_num': N,
            # 'map_size': [H, W],
            # 'init_grid_size': [H, W],
            # 'idx_token': idx_token,
            # 'agg_weight': agg_weight,
            'tokens_small_obj': tokens_small_obj,
            'tokens_to_discard': tokens_to_discard,
        }
        # not significance_score, small_scale_score but all predictions, fg_score, torch.exp(scale_score)
        return token_dict, fg_score_logit, scale_score_logit, valid_tokens
        # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]), probability (sum to 1 for each prediction of one token)

    def discard_split(self, x, mask=None):
        """
        scale guided supervision differentiable, but sumbel_softmax also differentiable
        we need to decouple these two processes to check their effects.
        Using predicted labels as selection results is also a good choice. I need to test this as well.
        Args:
            x:
            mask:

        Returns:

        """
        N, B, C = x.shape
        # token scoring, return N, B, Num_Classes (2 for fg, 3 for scale).
        fg_score_logit, scale_score_logit = self.token_scoring(x)
        # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]); torch.float32

        lsx = nn.LogSoftmax(dim=-1)  # exp(lsx) can get the softmax score, here we use lsx for gumbel_softmax
        fg_score, scale_score = lsx(fg_score_logit), lsx(scale_score_logit)

        # token sampling
        fg_mask = F.gumbel_softmax(fg_score, hard=True, dim=-1)  # 0 bg, 1 fg.
        # foreground score as significance_score
        significance_score = torch.exp(fg_score[:, :, -1])  # (N, B) torch.Size([650, 2])

        #  0: non-small (medium or large); 1: small.
        #  (old definition 0, small, 1, medium, 2, large is deprecated)
        scale_mask = F.gumbel_softmax(scale_score, hard=True, dim=-1)  # TODO: tau=1 set temperature
        small_scale_score = torch.exp(scale_score[:, :, -1])  # TODO: test fg_score * small_score

        # extract discard-tokens, small-object-tokens
        valid_tokens = get_valid_token_mask(x, mask).float()
        # to extract tokens inside image region (exclude padded regions)
        # tokens_to_discard = torch.logical_and(fg_mask[:, :, 0].bool(),
        #                                       valid_tokens)  # (N, B) num to discard: tensor([497., 363.]
        tokens_to_discard_original = fg_mask[:, :, 0] * valid_tokens  # (N, B) num to discard: tensor([497., 363.]
        # foreground and small objects, (N, B)
        tokens_small_obj_original = fg_mask[:, :, 1] * scale_mask[:, :, 1] * valid_tokens
        # tokens_large_obj = (fg_mask[:, :, 1] * scale_mask[:, :, 2]).bool()  # foreground + large objects
        # Generate new tokens from the old ones, filling the locations of tokens to discard with split tokens

        tokens_to_discard = tokens_to_discard_original.clone().detach().bool()
        # foreground and small objects, (N, B)
        tokens_small_obj = tokens_small_obj_original.clone().detach().bool()
        # calculate the number of tokens to split
        token_num = torch.stack([
            torch.sum(tokens_to_discard, dim=0),  # token to discard
            torch.sum(tokens_small_obj, dim=0),  # small object tokens
            # to control the maximum number of tokens to split
            torch.ones_like(torch.sum(tokens_to_discard, dim=0)) * self.max_split_token_num,
        ], dim=0)  # (3, B)
        #  (1, B), first column, the count for the first image; second column, second image.
        min_token_num = torch.min(token_num, dim=0)[0]  # 0, value; 1, indices

        # sample discard tokens if there are more tokens_to_discard than tokens_small_obj
        batch_ids = (token_num[0] - min_token_num > 0).nonzero(as_tuple=True)[0]
        # if (token_num[0] > min_token_num).sum() > 0:
        for k in batch_ids:
            # significance_score[tokens_to_discard] will return a 1d vector and thus does not work
            # Returns the indices that sort a tensor along a given dimension
            # in ascending order (in default) by value.
            sorted_ids = torch.argsort(significance_score[tokens_to_discard[:, k], k], descending=True)  #
            ids = sorted_ids[min_token_num[k]:]
            # torch.nonzero return N,1 tensor
            tokens_to_stop_discard = torch.nonzero(tokens_to_discard[:, k]).squeeze(-1)[ids]
            # change its value from True to False to disable discarding
            tokens_to_discard[tokens_to_stop_discard, k] = False

        # if (token_num[1] > min_token_num).sum() > 0:
        batch_ids = (token_num[1] - min_token_num > 0).nonzero(as_tuple=True)[0]
        for k in batch_ids:  # TODO: put this to a function
            sorted_ids = torch.argsort(small_scale_score[tokens_small_obj[:, k], k], descending=True)  #
            ids = sorted_ids[min_token_num[k]:]

            tokens_to_stop_split = torch.nonzero(tokens_small_obj[:, k]).squeeze(-1)[ids]
            tokens_small_obj[tokens_to_stop_split, k] = False

        x_small = torch.zeros_like(x)
        for k in range(B):
            # different image has different number of tokens to split or merge, thus
            # batch processing is not possible.
            img_small_obj_ids = tokens_small_obj[:, k]
            img_discard_token_ids = tokens_to_discard[:, k]
            x_k = x[:, k, :][img_small_obj_ids]  # M, C, where M is the number of tokens to split
            tokens_small_obj_new = self.token_split_conv(x_k)
            # # x[img_small_obj_ids, k, :] = tokens_small_obj_new[0]
            # # x[img_discard_token_ids, k, :] = tokens_small_obj_new[1]

            x_small[img_small_obj_ids, k, :] += tokens_small_obj_new[0]
            x_small[img_discard_token_ids, k, :] += tokens_small_obj_new[1]
        # # TODO: directly modify the value, differentiable? to check, maybe it is differentiable
        keep_mask = 1 - (tokens_to_discard_original * tokens_to_discard + tokens_small_obj_original * tokens_small_obj)
        x_new = x.clone() * keep_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) + x_small  # t

        token_dict = {
            'x': x_new,
            # 'token_num': N,
            # 'map_size': [H, W],
            # 'init_grid_size': [H, W],
            # 'idx_token': idx_token,
            # 'agg_weight': agg_weight,
            'tokens_small_obj': tokens_small_obj,
            'tokens_to_discard': tokens_to_discard,

            'fg_mask': fg_mask,
            'fg_obj_score': significance_score,
            'tokens_to_discard_original': tokens_to_discard_original,

            'small_scale_score': small_scale_score,
            'scale_mask': scale_mask,
            'tokens_small_obj_original': tokens_small_obj_original,
        }
        # not significance_score, small_scale_score but all predictions, fg_score, torch.exp(scale_score)
        return token_dict, fg_score_logit, scale_score_logit, valid_tokens
        # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]), probability (sum to 1 for each prediction of one token)
