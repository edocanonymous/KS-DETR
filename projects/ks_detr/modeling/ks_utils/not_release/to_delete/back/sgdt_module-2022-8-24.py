import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sgdt.sgdn_components import MIN_FG_SIGNIFICANCE, MAX_BG_SIGNIFICANCE

# --------------- from DynamicViT dyconvnext.py without modification
from models.sgdt.token_scoring import TokenScoringV1, TokenScoring, \
    TokenRemoveScoringWGF, TokenScoringWGFDynamicViT, TokenScoringConv
import numpy as np

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


def is_str_exist(full_str, sub_str):
    return full_str.find(sub_str) > -1


class TokenScoringConfigParser:
    def __init__(self, token_scoring_discard_split_criterion):
        self.token_scoring_discard_split_criterion = token_scoring_discard_split_criterion

    def _str_exist(self, sub_str):
        return is_str_exist(self.token_scoring_discard_split_criterion, sub_str)

    def _str_list_exist(self, sub_str_list):
        assert isinstance(sub_str_list, list)
        return [self._str_exist(sub_str) for sub_str in sub_str_list]

    @property
    def reclaim_padded_region(self):  # padded tokens, default: not claim
        return self._str_exist('reclaim_padded_region')

    @property
    def no_bg_remove(self):  # bg tokens, default: remove
        return self._str_exist('no_bg_remove')

    @property
    def no_split(self):  # fg tokens, default: split
        return self._str_exist('no_split')



    @property
    def ignore_bg_label(self):
        return self._str_exist('ignore_bg_label')



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

    def token_scoring_with_fg_pred_(self):
        """
        Making predictions or not
        if  token_scoring_discard_split_criterion in ['v0_with_gt',
                                 'v0_with_gt_and_reclaim_padded_region',
                                 'v0_with_gt_only_reclaim_padded_region'
                                          ]):
             return True

        Returns:


        """
        if self.token_scoring_discard_split_criterion in [
            'v1_selection_differentiable', # 'test_but_scoring_grad_for_loss_only'
        ]:
            return True
        else:
            return False

    def token_split_(self):
        """
        Making predictions or not
        if  token_scoring_discard_split_criterion in ['v0_with_gt',
                                 'v0_with_gt_and_reclaim_padded_region',
                                 'v0_with_gt_only_reclaim_padded_region'
                                          ]):
             return True

        Returns:


        """
        if self.token_scoring_discard_split_criterion in [
            'v0_with_gt_only_remove', 'pred_significance_all_fg_w_priority_only_remove',
        ]:
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
    def __init__(self, embed_dim, max_split_token_num=10000,
                 token_scoring_discard_split_criterion=None,
                 ):  # no restriction with 10000
        super().__init__()

        self.token_scoring_discard_split_criterion = token_scoring_discard_split_criterion
        self.token_scoring_config_parser = TokenScoringConfigParser(
            token_scoring_discard_split_criterion)

        # 'dynamic_vit_pred_fg_only_remove_0.1'
        if token_scoring_discard_split_criterion.find('dynamic_vit_pred_fg_only_remove') > -1:
            self.token_scoring = TokenScoringWGFDynamicViT(embed_dim=embed_dim)
            self.token_split_conv = None
        elif token_scoring_discard_split_criterion.find('conv_pred_fg_only_remove') > -1:
            self.token_scoring = TokenScoringConv(embed_dim=embed_dim)
            self.token_split_conv = None

        elif token_scoring_discard_split_criterion.find('pred_fg_only_remove') > -1:
            self.token_scoring = TokenRemoveScoringWGF(embed_dim=embed_dim)
            self.token_split_conv = None

        else:

            if token_scoring_discard_split_criterion is not None and \
                    self.token_scoring_config_parser.pred_score_():
                if self.token_scoring_config_parser.token_scoring_with_fg_pred_():
                    self.token_scoring = TokenScoringV1(embed_dim=embed_dim)
                elif token_scoring_discard_split_criterion == 'v1_selection_differentiable_sig_value':
                    self.token_scoring = TokenScoringV1(embed_dim=embed_dim)
                else:
                    self.token_scoring = TokenScoring(embed_dim=embed_dim)
            else:
                self.token_scoring = None

            if self.token_scoring_config_parser.token_split_():
                self.token_split_conv = TokenSplit(embed_dim=embed_dim)
            else:
                self.token_split_conv = None

        self.max_split_token_num = max_split_token_num

    def _pred_processing(self,
                         x, tokens_to_split_original, tokens_to_discard_original,
                         fg_score,
                         small_scale_score,
                         mask=None,
                         reclaim_padded_region=False,
                         no_bg_token_remove=False,
                         no_token_split=False,
                         ):
        assert ~no_token_split or (no_token_split and ~no_bg_token_remove), \
            'valid setting: 1) with token split 2), no token split, but with bg token remove'

        valid_tokens_float = get_valid_token_mask(x, mask).float()
        invalid_tokens = ~get_valid_token_mask(x, mask)

        src_mask_reclaimed = None
        if no_token_split:  # only_remove_bg_token_no_split
            # check if the setting is valid.
            assert (not no_bg_token_remove) and (not reclaim_padded_region), \
                'Valid setting for no_token_split: reclaim_padded_region False and ' \
                'with_bg_token_remove True (no_bg_token_remove False), but got ' \
                f'reclaim_padded_region = {reclaim_padded_region}, no_bg_token_remove = {no_bg_token_remove}'

            # update x_new by zero-outing the bg locations.
            x_new = x * (1 - tokens_to_discard_original).unsqueeze(-1)
            # 1 means fg locations, 0 means bg locations

            # update the attention mask

            src_mask_reclaimed = self.get_attn_mask_for_masking_out_removed_tokens(
                invalid_tokens=invalid_tokens, removed_tokens_mask=tokens_to_discard_original.bool())

            # No token to split
            tokens_to_split = tokens_to_split_original = torch.full_like(invalid_tokens, False)
            # save tokens to remove
            tokens_to_discard = tokens_to_discard_original.bool()

        else:  # with split
            # -----------------------------------
            # handling default settings: remove bg tokens and split fg tokens
            # -----------------------------------

            # check if the setting is valid.
            assert reclaim_padded_region or (not no_bg_token_remove), \
                'Valid setting for with_token_split: reclaim_padded_region True or with_bg_token_remove True ' \
                '(no_bg_token_remove False), but got ' \
                f'reclaim_padded_region = {reclaim_padded_region}, no_bg_token_remove = {no_bg_token_remove}'

            if reclaim_padded_region:
                if not no_bg_token_remove:  # predicted bg for valid tokens + padded region
                    # TODO: check if detach is needed
                    tokens_to_discard_original = (tokens_to_discard_original + invalid_tokens).bool().float()  # .detach().clone()

                    # update significance_score so that tokens for padded regions has high priority of being
                    # sampled
                    fg_score[invalid_tokens] = float("inf")

                else:  # 2) only padded region
                    # no need to update the score
                    tokens_to_discard_original = 1 - valid_tokens_float  # .detach().clone()
            # else:
            #     tokens_to_discard_original = tokens_to_discard_original  # (N, B)

            # assert not(torch.logical_and(tokens_to_discard_original, tokens_to_split_original).any()), \
            #     'There should be no overlap for tokens_to_discard_original and tokens_to_split'

            assert (tokens_to_discard_original * tokens_to_split_original).sum() == 0, \
                'There should be no overlap for tokens_to_discard_original and tokens_to_split'

            x_new, tokens_to_discard, tokens_to_split = self._extract_adapted_token(
                x=x, tokens_to_discard_original=tokens_to_discard_original,
                tokens_to_split_original=tokens_to_split_original,
                fg_score=fg_score,
                small_scale_score=small_scale_score,
                # TODO: change to other criteria for deciding which tokens to remove and split
            )

            if reclaim_padded_region:
                src_mask_reclaimed = self.get_attn_mask_for_claimed_padded_regions(
                    invalid_tokens=invalid_tokens, tokens_to_discard=tokens_to_discard)

        output_dict = {
            'x': x_new,
            'src_mask_reclaimed': src_mask_reclaimed,
            'tokens_small_obj': tokens_to_split,
            'tokens_to_discard': tokens_to_discard,
            'valid_tokens': valid_tokens_float,  # TODO: change this to bool()

            # for visualization only
            'tokens_to_discard_original': tokens_to_discard_original,
            'tokens_to_split_original': tokens_to_split_original,
            'significance_score': fg_score,

            # 'fg_score_logits': None,
            # 'small_scale_score': None,
            # 'fg_score': None,
            # fg_score, scale_score  # N, B, C, where C is the number of classes,
            # e.g., torch.Size([650, 2, 2]), torch.Size([630, 2, 3]);
            # 'fg_mask': fg_mask,
            # 'fg_obj_score': significance_score,
            # 'small_scale_score': significance_score,
            # 'scale_mask': fg_mask,
        }

        # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]), probability (sum to 1 for each prediction of one token)
        # return token_dict, fg_score_logit, scale_score_logit, valid_tokens_float
        return output_dict

    def discard_split_significance_all_fg_w_priority(
            self, x, mask=None,
            reclaim_padded_region=False,
            ignore_bg_label=False,
            only_remove_bg_token_no_split=False,
            min_fg_significance_thd=0.7,
            max_bg_significance_thd=0.3,
    ):
        """
        scale guided supervision differentiable, but sumbel_softmax also differentiable
        we need to decouple these two processes to check their effects.
        Using predicted labels as selection results is also a good choice. I need to test this as well.
        Args:
            only_remove_bg_token_no_split:
            min_fg_significance_thd:
            max_bg_significance_thd:
            ignore_bg_label: for testing the idea of only using the padded regions
            reclaim_padded_region:
            x:  N, B, C = x.shape
            mask:

        Returns:

        """
        # (N, B, 1) -> (N, B) e.g., torch.Size([572, 2, 1])
        significance_score_logit = self.token_scoring(x).squeeze(-1)  #
        significance_score = F.sigmoid(significance_score_logit).detach().clone()  # torch.Size([572, 2, 1])

        valid_tokens_float = get_valid_token_mask(x, mask).float()

        # token sampling, we only extract the most certain predictions
        # 0.5 or MIN_FG_SIGNIFICANCE
        # We never modify the token to split even if we reclaim_padded_region
        tokens_to_split_original = (significance_score >= min_fg_significance_thd).float() * valid_tokens_float

        # --------------------------- deprecated version
        # no need MAX_BG_SIGNIFICANCE as the tokens used as bg is selected from highly confident ones?
        # bg_mask = 1 - fg_mask
        # assert bg_mask[invalid_tokens].sum() == invalid_tokens.sum(), \
        #     'All invalid_tokens should be regarded as bg, no any invalid_tokens should be fg.'
        # ---------------------------

        # locations of invalid tokens and tokens with small significance values are excluded.
        # fg_mask + bg_mask < all tokens.
        tokens_to_discard_original = (significance_score < max_bg_significance_thd).float() * valid_tokens_float

        output_dict = self._pred_processing(
                         x=x, tokens_to_split_original=tokens_to_split_original,
                         tokens_to_discard_original=tokens_to_discard_original,
                         fg_score=significance_score,
                         small_scale_score=significance_score,
                         mask=mask,
                         reclaim_padded_region=reclaim_padded_region,
                         no_bg_token_remove=ignore_bg_label,
                         no_token_split=only_remove_bg_token_no_split,
                         )
        output_dict.update({
            'small_scale_score_logits': significance_score_logit})

        # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]), probability (sum to 1 for each prediction of one token)
        # return token_dict, fg_score_logit, scale_score_logit, valid_tokens_float
        return output_dict

    def discard_split_pred_fg(
            self, x, mask=None,  #
            reclaim_padded_region=False,
            ignore_bg_label=False,
            only_remove_bg_token_no_split=False,
            min_fg_significance_thd=None,  # 0.7,
            max_bg_significance_thd=None,  # 0.3,
            feat_map_size=None,
            with_global_feat=None,
            sgdt_targets=None,
            debug=False,
            inverse_bg_thd=False,
    ):

        # (N, B, 1) -> (N, B) e.g., torch.Size([572, 2, 1])
        if feat_map_size is not None or with_global_feat is not None:
            # assert with_global_feat is not None
            fg_score_logits = self.token_scoring(x, feat_map_size=feat_map_size,
                                                 with_global_feat=with_global_feat,
                                                 mask=mask)
        else:
            fg_score_logits = self.token_scoring(x)  #
            # extract the score of being fg tokens
        significance_score = F.softmax(fg_score_logits, dim=-1)[:, :, -1].detach().clone()  # torch.Size([572, 2, 1])

        # The following lines are same with
        valid_tokens_float = get_valid_token_mask(x, mask).float()

        if min_fg_significance_thd is None:
            min_fg_significance_thd = MIN_FG_SIGNIFICANCE
        if max_bg_significance_thd is None:
            max_bg_significance_thd = MAX_BG_SIGNIFICANCE
        tokens_to_split_original = (significance_score >= min_fg_significance_thd).float() * valid_tokens_float

        if not inverse_bg_thd:  # Normal setting
            tokens_to_discard_original = (significance_score < max_bg_significance_thd).float() * valid_tokens_float
        else:  # for debugging only
            tokens_to_discard_original = (significance_score > max_bg_significance_thd).float() * valid_tokens_float

        # filter out false bg predictions (tokens are fg but predicted bg)
        if debug:
            # fg_gt = torch.where(sgdt_targets['scale_gt'] > 0, 1, 0)  # torch.int64
            fg_gt = sgdt_targets['fg_gt'].bool()
            tokens_to_discard_original[fg_gt] = False  # fg tokens will not be discard.

        output_dict = self._pred_processing(
            x=x, tokens_to_split_original=tokens_to_split_original,
            tokens_to_discard_original=tokens_to_discard_original,
            fg_score=significance_score,
            small_scale_score=significance_score,
            mask=mask,
            reclaim_padded_region=reclaim_padded_region,
            no_bg_token_remove=ignore_bg_label,
            no_token_split=only_remove_bg_token_no_split,
        )
        output_dict.update({
            'fg_score_logits': fg_score_logits})

        # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]), probability (sum to 1 for each prediction of one token)
        # return token_dict, fg_score_logit, scale_score_logit, valid_tokens_float
        return output_dict

    def discard_split_pred_significance_all_fg_w_priority_differentiable(
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
        # -------------------------------------
        # significance_score = F.sigmoid(significance_score_logit).detach().clone()  # torch.Size([572, 2, 1])
        significance_score = F.sigmoid(significance_score_logit) # torch.Size([572, 2, 1])
        # -------------------------------------


        # token sampling gumbel_softmax
        valid_tokens_float = get_valid_token_mask(x, mask).float()
        fg_mask = (significance_score >= MIN_FG_SIGNIFICANCE).float() * valid_tokens_float  # 0.5 or MIN_FG_SIGNIFICANCE
        # no need MAX_BG_SIGNIFICANCE, as the tokens used as bg is selected from highly confident ones.
        bg_mask = 1 - fg_mask
        # bg_mask = (significance_score < MAX_BG_SIGNIFICANCE).float()

        invalid_tokens = ~get_valid_token_mask(x, mask)

        # background tokens
        tokens_to_discard_original = bg_mask * valid_tokens_float  # (N, B)
        tokens_to_split_original = fg_mask

        if reclaim_padded_region:
            if not ignore_bg_label:  # predicted bg for valid tokens + padded region
                # TODO: check if detach is needed
                tokens_to_discard_original = (bg_mask + invalid_tokens).bool().float()  # .detach().clone()

                # update significance_score so that tokens for padded regions has low scores (low priority of being
                # sampled. (later we can change this to high).
                significance_score[invalid_tokens] = float("-inf")  # -1e6  # set the score to be very lower

            else:  # 2) padded region only
                tokens_to_discard_original = invalid_tokens.float()  # .detach().clone()

        # assert not(torch.logical_and(tokens_to_discard_original, tokens_to_split_original).any()), \
        #     'There should be no overlap for tokens_to_discard_original and tokens_to_split'

        assert (tokens_to_discard_original * tokens_to_split_original).sum() == 0, \
            'There should be no overlap for tokens_to_discard_original and tokens_to_split'

        assert bg_mask[invalid_tokens].sum() == invalid_tokens.sum(), \
            'All invalid_tokens should be regarded as bg, no any invalid_tokens should be fg.'

        x_new, tokens_to_discard, tokens_to_split = self._extract_adapted_token(
            x=x, tokens_to_discard_original=tokens_to_discard_original,
            tokens_to_split_original=tokens_to_split_original,
            fg_score=significance_score,
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
            'tokens_to_split_original': tokens_to_split_original,
        }

        fg_score_logit = None
        scale_score_logit = significance_score_logit
        # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]), probability (sum to 1 for each prediction of one token)
        return token_dict, fg_score_logit, scale_score_logit, valid_tokens_float

    def _extract_adapted_token(self, x, tokens_to_discard_original,
                               tokens_to_split_original, fg_score,
                               small_scale_score, remove_split_criteria='min_value'
                               ):
        """
        the number of tokens to split and remove can be controlled by tokens_to_discard_original and
        tokens_to_split_original, the priority of sampling can be decided by the significance_score or
        small_scale_score.

        Args:
            x:
            tokens_to_discard_original: float
            tokens_to_split_original: float
            fg_score: lower score (less important bg tokens), higher priority of being sampled.
            small_scale_score: higher score, higher priority of being sampled.
            remove_split_criteria:
                'min_value', min(num_to_split, num_to_remove) for each image
                'max_remove':
        Returns:

        """
        # N, B, C = x.shape
        assert remove_split_criteria in ['min_value', 'max_remove']

        assert remove_split_criteria != 'max_remove', 'max_remove is not supported yet.'

        tokens_to_discard = tokens_to_discard_original.clone().detach().bool()
        # foreground and small objects, (N, B)
        tokens_small_obj = tokens_to_split_original.clone().detach().bool()

        # ======================================
        # calculate the number of tokens to split and remove so that number of split = number of remove
        # ======================================
        token_num = torch.stack([
            torch.sum(tokens_to_discard, dim=0),  # tokens to discard
            torch.sum(tokens_small_obj, dim=0),  # tokens to split
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
            sig_values = fg_score[tokens_to_discard[:, k], k]

            # only keep the first min_token_num[k] tokens that with SMALLEST significance values
            sorted_ids = torch.argsort(sig_values, descending=False)
            stop_ids = sorted_ids[min_token_num[k]:]

            # torch.equal(significance_score[torch.nonzero(tokens_to_discard[:, k]).squeeze(-1), k], sig_values) = True
            # torch.nonzero return N,1 tensor
            tokens_to_stop_discard = torch.nonzero(tokens_to_discard[:, k]).squeeze(-1)[stop_ids]
            # change its value from True to False to disable discarding
            tokens_to_discard[tokens_to_stop_discard, k] = False

        # if (token_num[1] > min_token_num).sum() > 0:
        batch_ids = (token_num[1] - min_token_num > 0).nonzero(as_tuple=True)[0]
        for k in batch_ids:
            # only keep the first min_token_num[k] tokens that with LARGEST significance values
            sorted_ids = torch.argsort(small_scale_score[tokens_small_obj[:, k], k], descending=True)  #
            stop_ids = sorted_ids[min_token_num[k]:]

            tokens_to_stop_split = torch.nonzero(tokens_small_obj[:, k]).squeeze(-1)[stop_ids]
            tokens_small_obj[tokens_to_stop_split, k] = False

        # Make sure there is no any token to be both removed and split
        assert not torch.any(torch.logical_and(tokens_small_obj, tokens_to_discard))

        x_new = self._reassemble_tokens(
            x=x, tokens_to_discard_original=tokens_to_discard_original,
            tokens_to_split_original=tokens_to_split_original,
            tokens_small_obj=tokens_small_obj, tokens_to_discard=tokens_to_discard)

        return x_new, tokens_to_discard, tokens_small_obj

    def _reassemble_tokens(self, x, tokens_to_discard_original, tokens_to_split_original,
                           tokens_small_obj, tokens_to_discard):
        """

        Args:
            x:
            tokens_to_discard_original: float()
            tokens_to_split_original:  float()
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
                tokens_to_discard_original * tokens_to_discard + tokens_to_split_original * tokens_small_obj)
        x_new = x.clone() * keep_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) + x_small  # t
        return x_new

    def _reassemble_tokens_v0(self, x, tokens_to_discard_original, tokens_to_split_original,
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
    def get_attn_mask_for_masking_out_removed_tokens(invalid_tokens, removed_tokens_mask):
        """

        Args:
            removed_tokens_mask:  (N, B), bool, True means locations to be removed, False, locations to keep.
            invalid_tokens: (N, B), bool
        Returns:

        """
        assert invalid_tokens.dtype == torch.bool and removed_tokens_mask.dtype == torch.bool
        assert ~(torch.logical_and(invalid_tokens, removed_tokens_mask).any()), \
            'No overlap should occur for tokens to remove and tokens invalid (padded tokens). '

        N, B = removed_tokens_mask.shape
        final_invalid_tokens = torch.logical_or(invalid_tokens, removed_tokens_mask)

        src_mask_reclaimed = torch.full((N, B), False, device=invalid_tokens.device)
        src_mask_reclaimed[final_invalid_tokens] = True
        src_mask_reclaimed = src_mask_reclaimed.permute(1, 0)  # N, B ->  B, N

        # We cannot mask all tokens as invalid.
        for k, src_mask in enumerate(src_mask_reclaimed):  # B, N
            if src_mask.all():
                inds = removed_tokens_mask[:, k].nonzero(as_tuple=True)[0]
                # set the first item to be False, to set this location to be valid.
                src_mask_reclaimed[k, inds[0]] = False

        return src_mask_reclaimed

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
                                          only_remove_bg_token_no_split=False,
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

        # small_scale_score = torch.randn_like(torch.exp(scale_score[:, :, -1]))
        # TODO: test fg_score * small_score
        significance_score = small_scale_score = sgdt_targets['scale_gt'] + \
                                                 1e-6 * torch.rand_like(sgdt_targets['scale_gt'])
        # to extract tokens inside image region (exclude padded regions)
        # tokens_to_discard = torch.logical_and(fg_mask[:, :, 0].bool(),
        #                                       valid_tokens)  # (N, B) num to discard: tensor([497., 363.]
        # the invalid tokens are marked as bg in the fg_mask
        # tokens_to_discard_original = fg_mask[:, :, 0] * valid_tokens  # (N, B) num to discard: tensor([497., 363.]
        valid_tokens_float = get_valid_token_mask(x, mask).float()
        invalid_tokens = ~(get_valid_token_mask(x, mask))
        assert fg_mask[:, :, 0][invalid_tokens].sum() == invalid_tokens.sum(), \
            'All invalid_tokens should be regarded as bg, no any invalid_tokens should be fg.'

        # extract discard-tokens, small-object-tokens
        src_mask_reclaimed = None
        if only_remove_bg_token_no_split:
            # update x_new by zero-outing the bg locations.
            x_new = x * (fg_mask[:, :, 1] * valid_tokens_float).unsqueeze(-1)
            # 1 means fg locations, 0 means bg locations

            # update the attention mask
            removed_tokens_mask = (fg_mask[:, :, 0] * valid_tokens_float).bool()
            src_mask_reclaimed = self.get_attn_mask_for_masking_out_removed_tokens(
                invalid_tokens=invalid_tokens, removed_tokens_mask=removed_tokens_mask)

            # for debugging only
            tokens_small_obj = tokens_to_split_original = torch.full_like(fg_gt, False)
            scale_mask = torch.full_like(scale_mask, False)
            tokens_to_discard_original = tokens_to_discard = removed_tokens_mask

        else:
            # foreground and small objects, (N, B)
            tokens_to_split_original = fg_mask[:, :, 1] * scale_mask[:, :, 1] * valid_tokens_float
            # tokens_large_obj = (fg_mask[:, :, 1] * scale_mask[:, :, 2]).bool()  # foreground + large objects
            # Generate new tokens from the old ones, filling the locations of tokens to discard with split tokens

            if not reclaim_padded_region:
                # to extract tokens inside image region (exclude padded regions
                tokens_to_discard_original = fg_mask[:, :, 0] * valid_tokens_float  # (N, B) num to discard: tensor([497., 363.]
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
                tokens_to_split_original=tokens_to_split_original,
                fg_score=significance_score,
                small_scale_score=small_scale_score
            )

            if reclaim_padded_region:
                src_mask_reclaimed = self.get_attn_mask_for_claimed_padded_regions(
                    invalid_tokens=invalid_tokens, tokens_to_discard=tokens_to_discard)

        # We cannot mask all tokens as invalid.
        for src_mask in src_mask_reclaimed:  # B, N
            assert not src_mask.all()

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
            'tokens_to_split_original': tokens_to_split_original,
        }
        # not significance_score, small_scale_score but all predictions, fg_score, torch.exp(scale_score)
        return token_dict, fg_score_logit, scale_score_logit, valid_tokens_float

    def discard_split_with_gt_token_score_but_pred_scoring(self, x, mask=None,
                                          reclaim_padded_region=False,
                                          sgdt_targets=None,
                                          ignore_bg_label=False,
                                          priority_setting=None,
                                          ):
        # ----------------
        # three options, 1) bg in gt, 2) padded region, 3) bg in gt + padded region
        # -----------------
        # scale_gt = fg_gt = torch.where(sgdt_targets['scale_gt'] > 0, 1.0, 0.0)  # ------------
        significance_score_logit = self.token_scoring(x).squeeze(-1)  #

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
        tokens_to_split_original = fg_mask[:, :, 1] * scale_mask[:, :, 1] * valid_tokens
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
            tokens_to_split_original=tokens_to_split_original,
            fg_score=significance_score,
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
            'tokens_to_split_original': tokens_to_split_original,
        }
        # not significance_score, small_scale_score but all predictions, fg_score, torch.exp(scale_score)
        # return token_dict, fg_score_logit, scale_score_logit, valid_tokens
        return token_dict, None, significance_score_logit, valid_tokens


    def discard_split(self, x, mask=None, scale_fg_independent=False):
        """
        scale guided supervision differentiable, but sumbel_softmax also differentiable
        we need to decouple these two processes to check their effects.
        Using predicted labels as selection results is also a good choice. I need to test this as well.
        Args:
            scale_fg_independent:
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
        invalid_tokens = ~(get_valid_token_mask(x, mask))
        # to extract tokens inside image region (exclude padded regions)
        # tokens_to_discard = torch.logical_and(fg_mask[:, :, 0].bool(),
        #                                       valid_tokens)  # (N, B) num to discard: tensor([497., 363.]

        # require_grad = True for tokens_to_discard_original and tokens_to_split_original
        tokens_to_discard_original_raw = fg_mask[:, :, 0] * valid_tokens  # (N, B) num to discard: tensor([497., 363.]
        # foreground and small objects, (N, B)

        if scale_fg_independent:
            # keep scale_mask and fg_mask independent to simplify the training task.
            tokens_to_split_original_raw = scale_mask[:, :, 1] * valid_tokens
        else:
            tokens_to_split_original_raw = fg_mask[:, :, 1] * scale_mask[:, :, 1] * valid_tokens

        # The tokens both to split or remove.
        ambiguous_tokens = tokens_to_discard_original_raw * tokens_to_split_original_raw

        # Remove the ambiguous_tokens from the remove set and split set.
        tokens_to_discard_original = tokens_to_split_original_raw - ambiguous_tokens
        tokens_to_split_original = tokens_to_split_original_raw - ambiguous_tokens

        tokens_to_discard = tokens_to_discard_original.clone().detach().bool()
        # foreground and small objects, (N, B)
        tokens_small_obj = tokens_to_split_original.clone().detach().bool()
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
        keep_mask = 1 - (tokens_to_discard_original * tokens_to_discard + tokens_to_split_original * tokens_small_obj)
        x_new = x.clone() * keep_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) + x_small  # t

        token_dict = {
            'x': x_new,

            'src_mask_reclaimed': None,
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
            'tokens_to_split_original': tokens_to_split_original,
        }
        # not significance_score, small_scale_score but all predictions, fg_score, torch.exp(scale_score)
        return token_dict, fg_score_logit, scale_score_logit, valid_tokens
        # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]), probability (sum to 1 for each prediction of one token)

    def forward(self, x, mask=None,
                # reclaim_padded_region=False,
                sgdt_targets=None,  # only for debugging
                feat_map_size=None,
                ):
        """

        Args:
            x:
            mask:

        Returns:

        """

        # return self.discard_split(x, mask)

        # return self.discard_split_with_gt_token_score(x, mask, sgdt_targets=sgdt_targets)
        if self.token_scoring_discard_split_criterion.find('gt_exp') > -1:
            # default setting
            reclaim_padded_region = self.token_scoring_config_parser.reclaim_padded_region
            ignore_bg_label = self.token_scoring_config_parser.ignore_bg_label
            only_remove_bg_token_no_split = False


            return self.discard_split_with_gt_token_score(
                x, mask, reclaim_padded_region=reclaim_padded_region,
                sgdt_targets=sgdt_targets,
                ignore_bg_label=ignore_bg_label,
                only_remove_bg_token_no_split=only_remove_bg_token_no_split,
            )
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
                ignore_bg_label=True,
            )
        elif self.token_scoring_discard_split_criterion == 'v0_with_gt_only_remove':
            return self.discard_split_with_gt_token_score(
                x, mask, reclaim_padded_region=False,
                sgdt_targets=sgdt_targets,
                ignore_bg_label=False,
                only_remove_bg_token_no_split=True,
            )
        elif self.token_scoring_discard_split_criterion == 'test_but_scoring_grad_for_loss_only':
            return self.discard_split_with_gt_token_score_but_pred_scoring(
                x, mask, reclaim_padded_region=False,
                sgdt_targets=sgdt_targets,
                ignore_bg_label=False
            )
            # return self.discard_split_with_gt_token_score_but_pred_scoring(
            #     x, mask, reclaim_padded_region=False,
            #     sgdt_targets=sgdt_targets,
            #     ignore_bg_label=False
            # )

        elif self.token_scoring_discard_split_criterion == 'v1_selection_differentiable':
            return self.discard_split(x, mask)
        elif self.token_scoring_discard_split_criterion == 'v1_selection_differentiable_sig_value':
            return self.discard_split_pred_significance_all_fg_w_priority_differentiable(x, mask)

        elif self.token_scoring_discard_split_criterion == \
                'pred_significance_all_fg_w_priority':
            return self.discard_split_significance_all_fg_w_priority(x, mask)
        elif self.token_scoring_discard_split_criterion == \
                'pred_significance_all_fg_w_priority_only_reclaim_padded_region':
            return self.discard_split_significance_all_fg_w_priority(
                x, mask,
                reclaim_padded_region=True,
                ignore_bg_label=True,
                only_remove_bg_token_no_split=False,
            )
        elif self.token_scoring_discard_split_criterion == \
                'pred_significance_all_fg_w_priority_only_remove':
            return self.discard_split_significance_all_fg_w_priority(
                x, mask,
                reclaim_padded_region=False,
                ignore_bg_label=False,
                only_remove_bg_token_no_split=True,
            )
        # 'pred_significance_all_fg_w_priority_and_reclaim_padded_region'
        elif self.token_scoring_discard_split_criterion == 'pred_significance_all_fg_bg_w_priority':
            return self.discard_split_significance_all_fg_w_priority(x, mask,
                    # min_fg_significance_thd=0.7,
                    # max_bg_significance_thd=0.3,
               )


        elif self.token_scoring_discard_split_criterion.find('fg_only_remove') > -1:
            discard_split_configs = self.token_scoring_discard_split_criterion.split('-')
            with_global_feat = np.array([v.find('w_gf') > -1 for v in discard_split_configs]).any()
            inverse_bg_thd = np.array([v.find('inverse') > -1 for v in discard_split_configs]).any()

            debug = np.array([v.find('debug') > -1 for v in discard_split_configs]).any()
            # sgdt_targets_new = sgdt_targets if debug else None

            max_bg_significance_thd = None
            for v in discard_split_configs:
                if v.find('bg_sig_thd') > -1:
                    max_bg_significance_thd = float(v.split('bg_sig_thd')[-1])
                    break

            return self.discard_split_pred_fg(
                x, mask,
                reclaim_padded_region=False,
                ignore_bg_label=False,
                only_remove_bg_token_no_split=True,
                max_bg_significance_thd=max_bg_significance_thd,
                inverse_bg_thd=inverse_bg_thd,
                with_global_feat=with_global_feat,
                feat_map_size=feat_map_size,
                sgdt_targets=sgdt_targets,
                debug=debug
            )
        else:
            raise NotImplementedError

