import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# --------------- from DynamicViT dyconvnext.py without modification
from models.sgdt.sgdt_components import SGDTConfigParse
# from models.sgdt.sgdt_ import SIGMA
from models.sgdt.token_scoring import TokenFGSmallScaleScoring, TokenFGScoringSigmoid, TokenFGScoringSoftmax, \
    TokenFGScoringWGF, TokenScoringWGFDynamicViT, TokenScoringConv, TokenScoringPredictorLG
from models.sgdt.topk import PerturbedTopKFunction, min_max_norm, HardTopK, extract_patches_from_indices

# FG_SIGNIFICANCE_THD = 0.6
# BG_SIGNIFICANCE_THD = 0.3  # only used in this file

FG_SIGNIFICANCE_THD = 0.0
BG_SIGNIFICANCE_THD = 1.0
EPSILON = 1e-5  # 1e-8


# _C.MVIT.DECAY_SIGMA = True
# _C.MVIT.SIGMA = 0.05


class TokenSplit(nn.Module):
    """ Importance Score (foreground score), Object scale Predictor
    """

    def __init__(self, embed_dim, expand=1):  # channel dim, also is the feature dimension for each token
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


class TokenScoringConfigParser(SGDTConfigParse):
    def __init__(self, token_scoring_discard_split_criterion):
        super().__init__(config_str=token_scoring_discard_split_criterion)

    @property
    def reclaim_padded_region(self):  # padded tokens, default: not claim
        return self.str_exist('reclaim_padded_region')

    @property
    def no_split(self):  # fg tokens, default: split
        return self.str_exist('no_split')

    @property
    def no_bg_token_remove(self):
        return self.str_exist('no_bg_token_remove')

    @property
    def predict_significance(self):
        # 'pred_significance_all_fg_w_priority':
        return self.str_exist('pred_significance')

    @property
    def with_global_feat(self):
        return self.str_exist('w_gf')

    @property
    def inverse_remove_thd(self):
        return self.str_exist('inverse_remove_thd')

    @property
    def inverse_split_thd(self):
        return self.str_exist('inverse_split_thd')

    @property
    def filter_false_remove(self):
        return self.str_exist('filter_false_remove')

    @property
    def filter_false_split(self):
        return self.str_exist('filter_false_split')

    @property
    def bg_sig_thd(self):
        discard_split_configs = self.config_str.split('-')
        bg_significance_thd = None
        for v in discard_split_configs:
            if v.find('bg_sig_thd') > -1:
                bg_significance_thd = float(v.split('bg_sig_thd')[-1])
                break
        return bg_significance_thd

    @property
    def split_sig_thd(self):
        discard_split_configs = self.config_str.split('-')
        split_significance_thd = None
        for v in discard_split_configs:
            if v.find('split_sig_thd') > -1:
                split_significance_thd = float(v.split('split_sig_thd')[-1])
                break
        return split_significance_thd

    @property
    def pred_score(self):
        """
        Making predictions or not
        if  token_scoring_discard_split_criterion in ['v0_with_gt',
                                 'v0_with_gt_and_reclaim_padded_region',
                                 'v0_with_gt_only_reclaim_padded_region'
                                          ]):
             return True

        Returns:


        """
        if self.config_str.find('gt_only_exp') > -1:
            return False
        else:
            return True

    def pred_fg_and_small_scale(self):
        """
        Making predictions or not
        if  token_scoring_discard_split_criterion in ['v0_with_gt',
                                 'v0_with_gt_and_reclaim_padded_region',
                                 'v0_with_gt_only_reclaim_padded_region'
                                          ]):
             return True

        Returns:


        """
        if self.config_str in [
            'v1_selection_differentiable',  # 'test_but_scoring_grad_for_loss_only'
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
        if self.config_str in [
            'v0_with_gt_only_remove', 'pred_significance_all_fg_w_priority_only_remove',
        ]:
            return False
        elif self.no_split:
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
    def __init__(self, embed_dim, max_split_token_num=10000,  # 10000  300
                 max_split_token_num_inference=10000,  # 300
                 token_scoring_discard_split_criterion=None,
                 ):  # no restriction with 10000
        super().__init__()
        assert token_scoring_discard_split_criterion is not None

        self.token_scoring_discard_split_criterion = token_scoring_discard_split_criterion
        self.token_scoring_config_parser = TokenScoringConfigParser(
            token_scoring_discard_split_criterion)

        # 'dynamic_vit_pred_fg_only_remove_0.1'
        if self.token_scoring_config_parser.pred_score:
            if self.token_scoring_config_parser.str_exist('pred_token_fg_dynamic_vit'):
                self.token_scoring = TokenScoringWGFDynamicViT(embed_dim=embed_dim)
            elif self.token_scoring_config_parser.str_exist('pred_token_fg_conv'):
                self.token_scoring = TokenScoringConv(embed_dim=embed_dim)
            elif self.token_scoring_config_parser.str_exist('pred_token_fg_wgf'):
                self.token_scoring = TokenFGScoringWGF(embed_dim=embed_dim)
            elif self.token_scoring_config_parser.str_exist('pred_token_fg'):
                self.token_scoring = TokenFGScoringSoftmax(embed_dim=embed_dim)
            elif self.token_scoring_config_parser.pred_fg_and_small_scale():
                self.token_scoring = TokenFGSmallScaleScoring(embed_dim=embed_dim)
            elif self.token_scoring_config_parser.predict_significance:
                self.token_scoring = TokenFGScoringSigmoid(embed_dim=embed_dim)

            elif self.token_scoring_config_parser.str_exist('topk_token_selection_differentiable'):
                # out_channels=1 for sigmoid use
                self.token_scoring = TokenScoringPredictorLG(embed_dim=embed_dim, out_channels=1)
            elif self.token_scoring_config_parser.str_exist('gumbel_softmax_token_selection_differentiable'):
                # out_channels=2 for softmax use
                self.token_scoring = TokenScoringPredictorLG(embed_dim=embed_dim, out_channels=2)
            else:
                raise NotImplementedError
        else:
            self.token_scoring = None

        if self.token_scoring_config_parser.token_split_():
            if self.token_scoring_config_parser.str_exist('split_2c'):
                self.token_split_conv = TokenSplit(embed_dim=embed_dim, expand=2)
                # For top-k method, we expand the channel by 1, otherwise, by 2.
            else:
                self.token_split_conv = TokenSplit(embed_dim=embed_dim, expand=1)
        else:
            self.token_split_conv = None

        self.max_split_token_num = max_split_token_num
        self.max_split_token_num_inference = max_split_token_num_inference
        self.num_samples = 100  # for top k token selection

    def _token_filtering_and_adaption(
            self, x,
            fg_score, small_scale_score,
            mask=None,
            reclaim_padded_region=False,  # token split, remove
            no_bg_token_remove=False,
            no_token_split=False,
            **kwargs,
    ):
        """
            sgdt_targets=None,  # using gt or not, only for debug purpose
            inverse_bg_thd=kwargs.pop('inverse_bg_thd', False),
            debug=kwargs.pop('debug', False)
        Returns:
,
        """

        valid_tokens_float = get_valid_token_mask(x, mask).float()

        bg_sig_thd = kwargs.get('bg_sig_thd', None)
        if bg_sig_thd is None: bg_sig_thd = BG_SIGNIFICANCE_THD

        split_sig_thd = kwargs.get('split_sig_thd', None)
        if split_sig_thd is None: split_sig_thd = FG_SIGNIFICANCE_THD

        tokens_to_discard_original = (fg_score < bg_sig_thd).float() * valid_tokens_float
        tokens_to_split_original = (small_scale_score >= split_sig_thd).float() * valid_tokens_float

        # inverse for debugging only,
        if kwargs.get('inverse_remove_thd', False):
            tokens_to_discard_original = (fg_score > bg_sig_thd).float() * valid_tokens_float

        if kwargs.get('inverse_split_thd', False):
            tokens_to_split_original = (small_scale_score < split_sig_thd).float() * valid_tokens_float

        # filter out false bg predictions (tokens are fg but predicted bg)
        sgdt_targets = kwargs.get('sgdt_targets', None)
        if kwargs.get('filter_false_remove', False):
            assert sgdt_targets is not None
            fg_gt = sgdt_targets['fg_gt'].bool()
            # (not discard) or (fg_gt)
            tokens_to_discard_original[fg_gt] = 0.0  # fg tokens will not be discard.

        if kwargs.get('filter_false_split', False):
            assert sgdt_targets is not None
            scale_gt = torch.where(sgdt_targets['scale_gt'] > 0, 1, 0)  # torch.int64
            # and operation
            tokens_to_split_original *= scale_gt  # only real split tokens will be to split.

        return self._pred_processing(
            x=x, tokens_to_split_original=tokens_to_split_original,
            tokens_to_discard_original=tokens_to_discard_original,
            fg_score=fg_score,
            small_scale_score=small_scale_score,
            mask=mask,
            reclaim_padded_region=reclaim_padded_region,
            no_bg_token_remove=no_bg_token_remove,
            no_token_split=no_token_split,
        )

    @staticmethod
    def pad_img_tokens(x, mask, num_tokens, tokens_to_discard_original, tokens_to_split_original,
                       fg_score,
                       small_scale_score):
        """
        Args:
            x: dim: (N, B, C), where N is the number of tokens, B is the batch size,
            mask: (B, N), 0 valid locations, True padding locations.
            num_tokens:

        Returns:

        """
        N, B, C = x.shape
        device = x.device
        assert num_tokens > 0
        padded_x = x.new_tensor(torch.zeros((num_tokens, B, C))).to(device)
        padded_mask = mask.new_tensor(torch.ones((B, num_tokens))).to(device)  # True means padded tokens
        x = torch.cat([x, padded_x], dim=0)
        mask = torch.cat([mask, padded_mask], dim=-1)

        # N, B
        padded_zeros = tokens_to_discard_original.new_tensor(torch.zeros((num_tokens, B))).to(device)
        tokens_to_discard_original = torch.cat([tokens_to_discard_original, padded_zeros], dim=0)
        tokens_to_split_original = torch.cat([tokens_to_split_original, padded_zeros], dim=0)
        fg_score = torch.cat([fg_score, padded_zeros], dim=0)
        small_scale_score = torch.cat([small_scale_score, padded_zeros], dim=0)

        return x, mask, \
               tokens_to_discard_original, tokens_to_split_original, fg_score, small_scale_score

    def _pred_processing(self,
                         x, tokens_to_discard_original, tokens_to_split_original,
                         fg_score,
                         small_scale_score,
                         mask=None,
                         reclaim_padded_region=False,
                         no_bg_token_remove=False,
                         no_token_split=False,
                         reclaim_token_high_priority=True,
                         **kwargs,
                         ):
        assert ~no_token_split or (no_token_split and ~no_bg_token_remove), \
            'valid setting: 1) with token split 2), no token split, but with bg token remove'

        valid_tokens_original_float = get_valid_token_mask(x, mask).float()

        src_mask_reclaimed = None

        # tokens_to_split_original = N, B
        num_tokens_to_split = tokens_to_split_original.sum(dim=0)
        # number of padded regions should be taken into consideration for deciding the
        # final number of tokens to increase
        num_invalid_tokens = (1 - valid_tokens_original_float).sum(dim=0)
        max_num_tokens_to_split = (num_tokens_to_split - num_invalid_tokens).max().item()

        num_tokens_to_increase = self.token_scoring_config_parser.extract_thd('num_tokens_to_increase')
        increase_resolution = False
        if num_tokens_to_increase is not None and num_tokens_to_increase > 0:
            # the maximum of increase is less or equal to the number of split.
            num_tokens_to_increase = int(min(num_tokens_to_increase, max_num_tokens_to_split))
            if num_tokens_to_increase > 0:
                increase_resolution = True
                x, mask, tokens_to_discard_original, tokens_to_split_original, fg_score, small_scale_score = \
                    self.pad_img_tokens(x, mask,
                                        num_tokens=num_tokens_to_increase,
                                        # other things to be adapted
                                        tokens_to_discard_original=tokens_to_discard_original,
                                        tokens_to_split_original=tokens_to_discard_original,
                                        fg_score=fg_score,
                                        small_scale_score=small_scale_score,
                                        )
                src_mask_reclaimed = mask

        valid_tokens_float = get_valid_token_mask(x, mask).float()
        invalid_tokens = ~get_valid_token_mask(x, mask)

        if no_token_split:  # no_token_split
            # check if the setting is valid.
            assert (not no_bg_token_remove) and (not reclaim_padded_region), \
                'Valid setting for no_token_split: reclaim_padded_region False and ' \
                'with_bg_token_remove True (no_bg_token_remove False), but got ' \
                f'reclaim_padded_region = {reclaim_padded_region}, no_bg_token_remove = {no_bg_token_remove}'

            # update x_new by zero-outing the bg locations.
            x_new = x * (1 - tokens_to_discard_original).unsqueeze(-1)
            # 1 means fg locations, 0 means bg locations

            # update the attention mask
            # ----------- if all tokens are to be removed
            removed_tokens_mask = tokens_to_discard_original.bool()  # N, B
            all_remove_check = torch.logical_or(invalid_tokens, removed_tokens_mask).all(dim=0)
            for k, all_remove in enumerate(all_remove_check):
                if all_remove:
                    # set the first tokens to 'not remove'
                    removed_tokens_mask[0, k] = False
            # --------------------
            src_mask_reclaimed = self.get_attn_mask_for_masking_out_removed_tokens(
                invalid_tokens=invalid_tokens, removed_tokens_mask=removed_tokens_mask)

            # No token to split
            tokens_to_split = tokens_to_split_original = torch.full_like(invalid_tokens, False)
            tokens_to_discard = removed_tokens_mask

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
                    tokens_to_discard_original = (
                            tokens_to_discard_original + invalid_tokens).bool().float()  # .detach().clone()

                    # update significance_score so that tokens for padded regions has high priority of being
                    # sampled
                    if reclaim_token_high_priority is None or reclaim_token_high_priority:
                        fg_score[invalid_tokens] = float("-inf")
                    else:
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
                **kwargs,
                # TODO: change to other criteria for deciding which tokens to remove and split
            )
            if not self.token_scoring_config_parser.str_exist('token_num_no_limit'):
                assert tokens_to_discard.float().sum() == tokens_to_split.float().sum(), \
                    'The number of tokens to discard should be equal to the number of tokens to split.'

            if reclaim_padded_region:
                src_mask_reclaimed = self.get_attn_mask_for_claimed_padded_regions(
                    invalid_tokens=invalid_tokens, tokens_to_discard=tokens_to_discard)

        # We cannot mask all tokens as invalid.
        if src_mask_reclaimed is not None:
            for src_mask in src_mask_reclaimed:  # B, N
                assert not src_mask.all()

        output_dict = {
            'x': x_new,
            'src_mask_reclaimed': src_mask_reclaimed,
            'tokens_small_obj': tokens_to_split,
            'tokens_to_discard': tokens_to_discard,
            'valid_tokens': valid_tokens_float,  # TODO: change this to bool()

            # valid_tokens in the original size, this is only used for loss calculation.
            'valid_tokens_original': valid_tokens_original_float,
            'increase_resolution': increase_resolution,

            # for visualization only
            'tokens_to_discard_original': tokens_to_discard_original,
            'tokens_to_split_original': tokens_to_split_original,
            # 'significance_score': small_scale_score,  # small_scale_score  fg_score

            'fg_score': fg_score,
            'small_scale_score': small_scale_score,
        }

        # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]), probability (sum to 1 for each prediction of one token)
        # return token_dict, fg_score_logit, scale_score_logit, valid_tokens_float
        return output_dict

    def _extract_adapted_token(self, x, tokens_to_discard_original,
                               tokens_to_split_original, fg_score,
                               small_scale_score,
                               remove_split_criteria='min_value',
                               **kwargs,
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

        # foreground and small objects, (N, B)
        tokens_to_discard = tokens_to_discard_original.clone().detach().bool()
        tokens_small_obj = tokens_to_split_original.clone().detach().bool()

        # ======================================
        # calculate the number of tokens to split and remove so that number of split = number of remove
        # ======================================
        num_remove = torch.sum(tokens_to_discard, dim=0)  # tokens to discard
        num_split = torch.sum(tokens_small_obj, dim=0)  # tokens to split

        if self.token_scoring_config_parser.str_exist('token_num_no_limit'):
            num_remove = torch.ones_like(torch.sum(tokens_to_discard, dim=0)) * self.max_split_token_num
            num_split = torch.ones_like(torch.sum(tokens_to_discard, dim=0)) * self.max_split_token_num

        token_num = torch.stack([
            num_remove,
            num_split,
            # to control the maximum number of tokens to split
            torch.ones_like(torch.sum(tokens_to_discard, dim=0)) * self.max_split_token_num,
        ], dim=0)  # (3, B)
        #  (1, B), first column, the count for the first image; second column, second image.
        min_token_num = torch.min(token_num, dim=0)[0]  # 0, value; 1, indices

        # ======================================
        # set the mask for tokens to split and remove based on the decided number
        # ======================================
        # sample discard tokens if there are more tokens_to_discard than tokens_small_obj
        batch_ids = (num_remove - min_token_num > 0).nonzero(as_tuple=True)[0]
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
        batch_ids = (num_split - min_token_num > 0).nonzero(as_tuple=True)[0]
        for k in batch_ids:
            # only keep the first min_token_num[k] tokens that with LARGEST significance values
            sorted_ids = torch.argsort(small_scale_score[tokens_small_obj[:, k], k], descending=True)  #
            stop_ids = sorted_ids[min_token_num[k]:]

            tokens_to_stop_split = torch.nonzero(tokens_small_obj[:, k]).squeeze(-1)[stop_ids]
            tokens_small_obj[tokens_to_stop_split, k] = False

        # Make sure there is no any token to be both removed and split
        assert not torch.any(torch.logical_and(tokens_small_obj, tokens_to_discard))

        # debug_gt_split_ratio = kwargs.pop('debug_gt_split_ratio', None)
        # if debug_gt_split_ratio is not None:
        #     # assert 0 <= debug_gt_split_ratio <= 1
        #     # print(f'debug_gt_split_ratio = {debug_gt_split_ratio}')
        #     sampling_method = 'random_order'  # random_order  priority_order
        #     tokens_small_obj = self.disable_selected_tokens(
        #         small_scale_score=small_scale_score,
        #         tokens_to_split_original=tokens_small_obj,
        #         debug_gt_split_ratio=debug_gt_split_ratio,
        #         sampling_method=sampling_method  # random_order  priority_order
        #     )
        #     tokens_to_discard = self.disable_selected_tokens(
        #         small_scale_score=fg_score,
        #         tokens_to_split_original=tokens_to_discard,
        #         debug_gt_split_ratio=debug_gt_split_ratio,
        #         sampling_method=sampling_method  # random_order  priority_order
        #     )

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

            if self.token_split_conv.expand == 1:
                if self.token_scoring_config_parser.str_exist('debug_split_1c'):
                    x_small[img_small_obj_ids, k, :] += tokens_small_obj_new
                else:
                    x_small[img_discard_token_ids, k, :] += tokens_small_obj_new

            elif self.token_split_conv.expand == 2:
                x_small[img_small_obj_ids, k, :] += tokens_small_obj_new[0]  # '+' to make it differentiable
                x_small[img_discard_token_ids, k, :] += tokens_small_obj_new[1]
            else:
                raise NotImplementedError
                # x_small[img_small_obj_ids, k, :] += tokens_small_obj_new[0]  # '+' to make it differentiable

        if self.token_split_conv.expand == 1:

            if self.token_scoring_config_parser.str_exist('debug_split_1c'):
                # no claim padded mask, just conduct split_conv on the original tokens to split.
                keep_mask = 1 - (tokens_to_split_original * tokens_small_obj)
            else:
                keep_mask = 1 - (tokens_to_discard_original * tokens_to_discard)

            x_new = x.clone() * keep_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) + x_small  #
        elif self.token_split_conv.expand == 2:
            # TODO: check if logic_add is not differentiable?
            # old version for split token from c to 2C
            keep_mask = 1 - (
                    tokens_to_discard_original * tokens_to_discard + tokens_to_split_original * tokens_small_obj)
            x_new = x.clone() * keep_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) + x_small  # t

        else:
            raise NotImplementedError
        return x_new

    @staticmethod
    def disable_selected_tokens(tokens_to_split_original, small_scale_score,
                                debug_gt_split_ratio, sampling_method='random_order'
                                ):
        # random_order, priority_order, priority_order_inverse
        assert sampling_method in ['random_order', 'priority_order', 'priority_order_inverse']
        assert 0 <= debug_gt_split_ratio <= 1

        tokens_small_obj = tokens_to_split_original.detach().clone().bool()
        num_valids = torch.sum(tokens_to_split_original, dim=0)

        for k, num in enumerate(num_valids):

            sorted_ids = torch.argsort(small_scale_score[tokens_small_obj[:, k], k], descending=True)  #
            if sampling_method == 'random_order':
                inds = torch.randperm(sorted_ids.shape[0])[:int(num.item())].to(device=sorted_ids.device)
                sorted_ids = sorted_ids[inds]
            elif sampling_method == 'priority_order_inverse':
                sorted_ids = torch.argsort(small_scale_score[tokens_small_obj[:, k], k], descending=False)  #

            keep_num = int(num * debug_gt_split_ratio)
            stop_ids = sorted_ids[keep_num:]
            tokens_to_stop_split = torch.nonzero(tokens_small_obj[:, k]).squeeze(-1)[stop_ids]
            tokens_small_obj[tokens_to_stop_split, k] = 0.0
        return tokens_small_obj.float()

    def discard_split_with_gt_token_score(self, x, mask=None,
                                          sgdt_targets=None,
                                          reclaim_padded_region=False,
                                          no_token_split=False,
                                          no_bg_token_remove=False,
                                          reclaim_token_high_priority=None,
                                          **kwargs,
                                          ):
        fg_gt, scale_gt = sgdt_targets['fg_gt'], sgdt_targets['scale_gt']

        split_score = sgdt_targets['scale_gt'] + \
                      EPSILON * torch.rand_like(sgdt_targets['scale_gt'])
        fg_score = sgdt_targets['fg_gt'] + \
                   EPSILON * torch.rand_like(sgdt_targets['fg_gt'])

        # TODO: check why below exhibits low accuracy why?
        # split_score = fg_score = sgdt_targets['scale_gt'] + \
        #                          EPSILON * torch.rand_like(sgdt_targets['scale_gt'])
        # split_score = sgdt_targets['scale_gt']
        # fg_score = sgdt_targets['fg_gt']

        valid_tokens_float = get_valid_token_mask(x, mask).float()
        tokens_to_discard_original = (1 - fg_gt.float()) * valid_tokens_float
        tokens_to_split_original = (scale_gt > 0).float() * valid_tokens_float
        assert (tokens_to_discard_original * tokens_to_split_original).sum() == 0

        # ========================================
        # for debugging
        # ========================================
        debug_gt_split_ratio = self.token_scoring_config_parser.extract_thd(sub_str='debug_gt_split_ratio')
        debug_gt_remove_ratio = self.token_scoring_config_parser.extract_thd(sub_str='debug_gt_remove_ratio')

        if debug_gt_split_ratio is not None:
            # assert 0 <= debug_gt_split_ratio <= 1
            # print(f'debug_gt_split_ratio = {debug_gt_split_ratio}')
            if self.token_scoring_config_parser.str_exist(sub_str='debug_gt_split_sampling_priority'):
                sampling_method = 'priority_order'
            else:
                sampling_method = 'random_order'

            tokens_to_split_original = self.disable_selected_tokens(
                small_scale_score=split_score,
                tokens_to_split_original=tokens_to_split_original,
                debug_gt_split_ratio=debug_gt_split_ratio,
                sampling_method=sampling_method  # random_order  priority_order
            )

        if debug_gt_remove_ratio is not None:
            if self.token_scoring_config_parser.str_exist(sub_str='debug_gt_remove_sampling_random'):
                sampling_method = 'priority_order'
            else:
                sampling_method = 'random_order'

            tokens_to_discard_original = self.disable_selected_tokens(
                small_scale_score=fg_score,
                tokens_to_split_original=tokens_to_discard_original,
                debug_gt_split_ratio=debug_gt_remove_ratio,
                sampling_method=sampling_method  # random_order  priority_order
            )

        # if self.token_scoring_config_parser.str_exist(sub_str='use_proposal') or \
        #         self.token_scoring_config_parser.str_exist(sub_str='use_decoder_proposal'):
        use_proposal = 'proposal_fg_gt' in sgdt_targets or 'proposal_scale_gt' in sgdt_targets
        if use_proposal:
            # gt_ratio = self.token_scoring_config_parser.extract_thd(sub_str='gt_ratio')
            gt_ratio = kwargs.get('gt_ratio', None)
            random_value = random.uniform(0, 1)
            # Identify the cases when we update the proposal: 1) gt_ratio is None, 2)
            if gt_ratio is None or random_value > gt_ratio:
                proposal_fg_gt, proposal_scale_gt = sgdt_targets['proposal_fg_gt'], sgdt_targets['proposal_scale_gt']
                # TODO: check if I should use only a single score for the two.
                proposal_split_score = sgdt_targets['proposal_scale_gt'] + \
                                       EPSILON * torch.rand_like(sgdt_targets['proposal_scale_gt'])
                proposal_fg_score = sgdt_targets['proposal_fg_gt'] + \
                                    EPSILON * torch.rand_like(sgdt_targets['proposal_fg_gt'])

                assert (proposal_fg_gt > 0).sum() == (proposal_fg_gt == 1).sum(), \
                    'proposal_fg_gt only allow 0 or 1 values, but other values found.'

                proposal_tokens_to_discard_original = (1 - proposal_fg_gt.float()) * valid_tokens_float
                proposal_tokens_to_split_original = (proposal_scale_gt > 0).float() * valid_tokens_float

                # tokens can be both in discard and split
                # assert (proposal_tokens_to_discard_original * proposal_tokens_to_split_original).sum() == 0

                proposal_split_thd = self.token_scoring_config_parser.extract_thd(sub_str='proposal_split_thd')
                if proposal_split_thd is not None:
                    proposal_tokens_to_split_original = (proposal_scale_gt > proposal_split_thd).float() * \
                                                        valid_tokens_float

                # proposal_remove_thd = self.token_scoring_config_parser.extract_thd(sub_str='proposal_remove_thd')
                # if proposal_remove_thd is not None:
                #     raise NotImplementedError
                #     # proposal_tokens_to_discard_original = (1 - proposal_fg_gt.float()) * valid_tokens_float

                if self.token_scoring_config_parser.str_exist(sub_str='proposal_split_gt_filter'):
                    # only keep tokens in both proposal and gt.
                    proposal_tokens_to_split_original = tokens_to_split_original * proposal_tokens_to_split_original
                    # cnt = tokens_to_split_original.sum()
                    # new_cnt = tokens_to_split_original.sum()
                    # print(f'cnt / new_cnt = {cnt / new_cnt * 100}%')

                if self.token_scoring_config_parser.str_exist(sub_str='proposal_remove_gt_filter'):
                    proposal_tokens_to_discard_original = tokens_to_discard_original * proposal_tokens_to_discard_original

                tokens_to_split_original = proposal_tokens_to_split_original
                tokens_to_discard_original = proposal_tokens_to_discard_original

                if not self.token_scoring_config_parser.str_exist(sub_str='without_scoring_proposal'):
                    split_score = proposal_split_score
                    fg_score = proposal_fg_score

        output_dict = self._pred_processing(
            x=x, tokens_to_split_original=tokens_to_split_original,
            tokens_to_discard_original=tokens_to_discard_original,
            fg_score=fg_score,
            small_scale_score=split_score,
            mask=mask,
            reclaim_padded_region=reclaim_padded_region,
            no_bg_token_remove=no_bg_token_remove,
            no_token_split=no_token_split,
            **kwargs,
            # reclaim_token_high_priority=reclaim_token_high_priority
        )

        output_dict.update({
            'fg_score': fg_gt,
            'small_scale_score': scale_gt,
            'significance_score': scale_gt,
        })
        # if use_proposal:
        #     output_dict.update({
        #         'fg_score': fg_gt,
        #         'small_scale_score': fg_score,
        #         'significance_score': split_score,
        #     })
        # else:
        #     output_dict.update({
        #         'fg_score': fg_gt,
        #         'small_scale_score': scale_gt,
        #         'significance_score': scale_gt,
        #     }
        #     )

        # # fake logit for compatibility of later code.
        # fg_gt_fake = torch.where(sgdt_targets['fg_gt'] > 0, 1, 0)
        # scale_gt_fake = torch.where(sgdt_targets['scale_gt'] > 0, 1, 0)
        # fg_score_logit, scale_score_logit = F.one_hot(fg_gt_fake, num_classes=2).float(), \
        #                                     F.one_hot(scale_gt_fake, num_classes=2).float()
        # output_dict.update({
        #     'fg_score_logit': fg_score_logit,
        #     'small_scale_score_logit': scale_score_logit,
        # })
        # output_dict.update({
        #     # disable update the src_key_padding_mask
        #     'src_mask_reclaimed': None,
        #     # # disable update the pos encoding
        #     # 'tokens_small_obj': torch.zeros_like(output_dict['tokens_small_obj']).bool(),
        #     # 'tokens_to_discard': torch.zeros_like(output_dict['tokens_to_discard']).bool(),
        # })
        return output_dict

    def discard_split_with_gt_token_score_debug_split_1c(self, x, mask=None,
                                                         sgdt_targets=None,
                                                         reclaim_padded_region=False,
                                                         no_token_split=False,
                                                         no_bg_token_remove=False,
                                                         reclaim_token_high_priority=None,
                                                         **kwargs,
                                                         ):
        fg_gt, scale_gt = sgdt_targets['fg_gt'], sgdt_targets['scale_gt']

        # --------------------------------
        # TODO: check why below exhibits low accuracy why?
        split_score = sgdt_targets['scale_gt'] + \
                      EPSILON * torch.rand_like(sgdt_targets['scale_gt'])
        fg_score = sgdt_targets['fg_gt'] + \
                   EPSILON * torch.rand_like(sgdt_targets['fg_gt'])
        # ------------------------------------
        # split_score = fg_score = sgdt_targets['scale_gt'] + \
        #                          EPSILON * torch.rand_like(sgdt_targets['scale_gt'])
        # split_score = sgdt_targets['scale_gt']
        # fg_score = sgdt_targets['fg_gt']

        valid_tokens_float = get_valid_token_mask(x, mask).float()
        tokens_to_discard_original = (1 - fg_gt.float()) * valid_tokens_float
        tokens_to_split_original = (scale_gt > 0).float() * valid_tokens_float
        assert (tokens_to_discard_original * tokens_to_split_original).sum() == 0

        output_dict = self._pred_processing(
            x=x, tokens_to_split_original=tokens_to_split_original,
            tokens_to_discard_original=tokens_to_discard_original,
            fg_score=fg_score,
            small_scale_score=split_score,
            mask=mask,
            reclaim_padded_region=reclaim_padded_region,
            no_bg_token_remove=no_bg_token_remove,
            no_token_split=no_token_split,
            **kwargs,
        )

        output_dict.update({
            'fg_score': fg_gt,
            'small_scale_score': scale_gt,
            'significance_score': scale_gt,
        })

        # special adaption for debug purpose
        output_dict.update({
            # disable update the src_key_padding_mask
            'src_mask_reclaimed': mask,  # None (deprecated)
            # disable update the pos encoding.
            'tokens_small_obj': torch.zeros_like(output_dict['tokens_small_obj']).bool(),
            'tokens_to_discard': torch.zeros_like(output_dict['tokens_to_discard']).bool(),
        })
        return output_dict

    def discard_split_with_gt_token_score_but_pred_scoring(self, x, mask=None,
                                                           sgdt_targets=None,
                                                           reclaim_padded_region=False,
                                                           no_token_split=False,
                                                           no_bg_token_remove=False,
                                                           reclaim_token_high_priority=None,
                                                           **kwargs,
                                                           ):
        # GT based prediction
        output_dict = self.discard_split_with_gt_token_score(
            x=x, mask=mask,
            sgdt_targets=sgdt_targets,
            reclaim_padded_region=reclaim_padded_region,
            no_token_split=no_token_split,
            no_bg_token_remove=no_bg_token_remove,
            reclaim_token_high_priority=reclaim_token_high_priority,
            **kwargs,
        )

        # Prediction
        significance_score_logit = self.token_scoring(x).squeeze(-1)  # sigmoid not softmax
        # significance_score = F.sigmoid(significance_score_logit).detach().clone()  # torch.Size([572, 2, 1])

        output_dict.update({
            'small_scale_score_logit': significance_score_logit})
        return output_dict

    def discard_split_significance_all_fg_w_priority(
            self, x, mask=None,
            reclaim_padded_region=False,
            no_bg_token_remove=False,
            no_token_split=False,
            **kwargs
    ):
        """
        kwargs.pop('norm_layer', None)
        """
        # (N, B, 1) -> (N, B) e.g., torch.Size([572, 2, 1])
        significance_score_logit = self.token_scoring(x).squeeze(-1)  # sigmoid not softmax
        significance_score = F.sigmoid(significance_score_logit).detach().clone()  # torch.Size([572, 2, 1])

        output_dict = self._token_filtering_and_adaption(
            x=x, fg_score=significance_score,
            small_scale_score=significance_score,
            mask=mask,
            reclaim_padded_region=reclaim_padded_region,  # token split, remove
            no_bg_token_remove=no_bg_token_remove,
            no_token_split=no_token_split,
            **kwargs,  # sgdt_targets=sgdt_targets,
        )
        output_dict.update({
            'small_scale_score_logit': significance_score_logit})

        return output_dict

    def discard_split_pred_fg(
            self, x, mask=None,  #
            feat_map_size=None,
            with_global_feat=None,
            reclaim_padded_region=False,  # token split, remove
            no_bg_token_remove=False,
            no_token_split=False,
            **kwargs
    ):
        """
        **kwargs examples:
            split_sig_thd=split_sig_thd,  # 0.7, # token filtering
            bg_sig_thd=bg_sig_thd,  # 0.3,
            inverse_bg_thd=inverse_bg_thd,
        """

        # (N, B, 1) -> (N, B) e.g., torch.Size([572, 2, 1])
        if feat_map_size is not None or with_global_feat is not None:
            # assert with_global_feat is not None
            fg_score_logit = self.token_scoring(x, feat_map_size=feat_map_size,
                                                with_global_feat=with_global_feat,
                                                mask=mask)
        else:
            fg_score_logit = self.token_scoring(x)  #
            # extract the score of being fg tokens
        significance_score = F.softmax(fg_score_logit, dim=-1)[:, :, -1].detach().clone()  # torch.Size([572, 2, 1])

        # if kwargs.pop('split_only_ambiguous_token', False):
        if self.token_scoring_config_parser.str_exist('split_only_ambiguous_token'):
            significance_score = 0.5 - torch.abs(significance_score - 0.5)

        output_dict = self._token_filtering_and_adaption(
            x=x, fg_score=significance_score,
            small_scale_score=significance_score,
            mask=mask,
            reclaim_padded_region=reclaim_padded_region,  # token split, remove
            no_bg_token_remove=no_bg_token_remove,
            no_token_split=no_token_split,
            **kwargs,  # sgdt_targets=sgdt_targets,
        )
        output_dict.update(
            {'fg_score_logit': fg_score_logit}
        )

        # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]), probability (sum to 1 for each prediction of one token)
        # return token_dict, fg_score_logit, scale_score_logit, valid_tokens_float
        return output_dict

    def discard_split_topk_token_selection_differentiable(
            self, x, mask=None,  #
            sgdt_targets=None,
            feat_map_size=None,
            with_global_feat=None,
            reclaim_padded_region=False,  # token split, remove
            no_bg_token_remove=False,
            no_token_split=False,
            **kwargs
    ):
        """
        **kwargs examples:
            split_sig_thd=split_sig_thd,  # 0.7, # token filtering
            bg_sig_thd=bg_sig_thd,  # 0.3,
            inverse_bg_thd=inverse_bg_thd,
        """

        # global indicator
        valid_tokens_float = get_valid_token_mask(x, mask).float()  # N, B

        tokens_to_discard_original = 1 - valid_tokens_float  # .detach().clone()
        # tokens_to_split_original = valid_tokens_float

        significance_score_logit = self.token_scoring(x, feat_map_size=feat_map_size,
                                                      with_global_feat=with_global_feat,
                                                      mask=mask)

        # (N, B, 1) -> (N, B) e.g., torch.Size([572, 2, 1]), set the score of invalid tokens to 0, so that
        # they will have no chance to be selected.
        significance_score = torch.sigmoid(significance_score_logit).squeeze(-1) * valid_tokens_float

        sigma = kwargs.get('sigma', None)
        # if sigma is None:
        #     sigma = SIGMA

        N, B, C = x.shape

        tokens_small_obj = torch.zeros_like(valid_tokens_float)
        tokens_to_discard = torch.zeros_like(valid_tokens_float)
        invalid_token_num = torch.sum((1 - valid_tokens_float), dim=0).int()  # (N, B)

        final_split_num = 0
        # N, B, C  torch.Size([736, 2, 256])
        x_small = torch.zeros_like(x)  # to store updated split tokens.

        # different image has different number of tokens to split or merge, thus
        # batch processing is not possible.
        split_tok_indicators = []
        x_weight = torch.zeros_like(x)  # to store updated split tokens.
        for k in range(B):
            # ----------------------------
            if self.token_scoring_config_parser.str_exist('max_num_split'):
                # assume all tokens can be split (all tokens are fg tokens)
                top_k = int(min(valid_tokens_float[:, k].sum().item(),
                                invalid_token_num[k].item(),
                                self.max_split_token_num)
                            )
            elif self.token_scoring_config_parser.str_exist('best_num_split'):
                # use the gt top_k for preliminary experiments
                top_k = int(
                    min(sgdt_targets['fg_gt'][:, k].sum().item(),
                        invalid_token_num[k].item(),
                        self.max_split_token_num)
                )
            else:
                raise NotImplementedError
                # max_split_token_num = min((significance_score[:, k] > 0.5).sum().item(), self.max_split_token_num)
                # # tokens to split should be no more than number of valid tokens.
                # top_k = int(min(valid_tokens_float[:, k].sum().item(), invalid_token_num[k].item(), max_split_token_num))

            # ----------------------------

            # top_k = max(1, top_k)  # set at least choose top 1 to avoid top_k = 0 for all images in the batch
            final_split_num += top_k

            # We should never normalize the sigmoid values, but just the logit, Normalize the logits twice
            # will cause almost all the scores close to be 1.
            # scores = significance_score[:, k].unsqueeze(0)  # N, ->  1, N (to fit the b, N format)
            # scores = min_max_norm(scores)

            # # N, B, 1 -> 1, N
            scores = significance_score_logit[:, k, 0].unsqueeze(0)

            # change the scores of invalid token to be -1 so that they will not be selected.
            tmp_mask = valid_tokens_float + (1 - valid_tokens_float) * (-1)
            scores = min_max_norm(scores) * tmp_mask[:, k].unsqueeze(0)

            x_k = x[:, k, :].unsqueeze(0)  # N, C -> 1, N, C

            if self.training:
                # indicator is not int type, but float (e.g., 0.774),
                # There may be repeated element in top-k if we use argmax(), because it is soft top-k.

                # b k d : 1, top_k, N, e.g., torch.Size([1, 32, 736]) torch.float32
                indicator = PerturbedTopKFunction.apply(scores, top_k, self.num_samples, sigma)
                # indicator = einops.rearrange(indicator, "b k d -> b d k")  #
                # indicators = rearrange(indicators, "b d k -> b k d")
                patches = torch.einsum("b k d, b d c -> b k c", indicator, x_k)  # b, k, c, (1, top_k, c)

                # How to get the indices of the top k locations? Does PerturbedTopKFunction has the same
                #  top k location with HardTopK? No.
                # PerturbedTopKFunction returns soft top-k, not equal to the hard top k.
                # So using the indices of hard top k will cause the position embedding adaption error.
                indices = HardTopK(top_k, scores)  # # b, k; indices has been sorted in HardTopK.

                # Save the indicator for adapting the position encoding
                split_tok_indicators.append(indicator)
            else:
                # indices = self.get_indices(scores, 1)
                indices = HardTopK(top_k, scores)  # # b, k; indices has been sorted in HardTopK.
                patches = extract_patches_from_indices(x_k, indices)

            # print(f'patches.requires_grad = {patches.requires_grad}')
            indices = indices.squeeze(0)  # 1, top_k -> top_k

            # tokens_small_obj is only used for debugging, it will not attend the forward propagation.
            tokens_small_obj[:, k][indices] = 1
            img_small_obj_ids = tokens_small_obj[:, k].bool()
            assert not tokens_small_obj.requires_grad

            # print(f'x_k_new.requires_grad = {x_k_new.requires_grad}')
            # if valid_tokens_float[indices, k].sum() != indices.shape[0]:
            #     print(f'What happened?')
            assert valid_tokens_float[indices, k].sum() == indices.shape[0]

            x_k_new = patches.squeeze(0)  # top_k, c
            tokens_small_obj_new = self.token_split_conv(x_k_new)

            tokens_to_discard_inds = torch.nonzero(tokens_to_discard_original[:, k]).squeeze(-1)[:top_k]
            tokens_to_discard[tokens_to_discard_inds, k] = 1  # record the locations of tokens to discard for debugging
            img_discard_token_ids = tokens_to_discard[:, k].bool()

            # x_small[indices, k, :] = tokens_small_obj_new[0]
            if self.token_split_conv.expand == 1:

                if self.token_scoring_config_parser.str_exist('debug_split_1c'):
                    if self.training:
                        # patches = torch.einsum("b k d, b d c -> b k c", indicator, x_k)  # b, k, c, (1, top_k, c)
                        patches_inverse = torch.einsum("b d k, b k c -> b d c",
                                                       indicator.permute(0, 2, 1),  # b k d -> b d k
                                                       tokens_small_obj_new.unsqueeze(0)  # k, c -> b, k, c
                                                       )
                        x_small[:, k, :] += patches_inverse.squeeze(0)  # 1, d, c -> d, c (d = N)
                        weight_inverse = torch.einsum("b d k, b k c -> b d c",
                                                      indicator.permute(0, 2, 1),  # b k d -> b d k
                                                      torch.ones_like(tokens_small_obj_new.unsqueeze(0))
                                                      # k, c -> b, k, c
                                                      )
                        x_weight[:, k, :] = weight_inverse.squeeze(0)
                    else:
                        x_small[img_small_obj_ids, k, :] += tokens_small_obj_new
                else:
                    x_small[img_discard_token_ids, k, :] += tokens_small_obj_new
                # x_small[img_discard_token_ids, k, :] = tokens_small_obj_new

            elif self.token_split_conv.expand == 2:
                #
                # torch.Size([1, 736, 256])
                if self.training:
                    patches_inverse = torch.einsum("b d k, b k c -> b d c",
                                                   indicator.permute(0, 2, 1),  # b k d -> b d k
                                                   tokens_small_obj_new[0].unsqueeze(0)  # k, c -> b, k, c
                                                   )
                    x_small[:, k, :] += patches_inverse.squeeze(0)  # 1, d, c -> d, c (d = N)

                    weight_inverse = torch.einsum("b d k, b k c -> b d c",
                                                  indicator.permute(0, 2, 1),  # b k d -> b d k
                                                  torch.ones_like(tokens_small_obj_new[0].unsqueeze(0))
                                                  # k, c -> b, k, c
                                                  )
                    x_weight[:, k, :] = weight_inverse.squeeze(0)
                else:
                    x_small[img_small_obj_ids, k, :] += tokens_small_obj_new[0]  # '+' to make it differentiable

                # reclaimed regions
                x_small[img_discard_token_ids, k, :] += tokens_small_obj_new[1]
            else:
                raise NotImplementedError
                # x_small[img_small_obj_ids, k, :] += tokens_small_obj_new[0]  # '+' to make it differentiable

        # there should be no token in both tokens_small_obj and tokens_to_discard
        assert (tokens_to_discard * tokens_small_obj).max() < 1

        # TODO: update keep mask and pos encoding in Transformer.
        if self.token_split_conv.expand == 1:
            if self.token_scoring_config_parser.str_exist('debug_split_1c'):
                # no claim padded mask, just conduct split_conv on the original tokens to split.
                if self.training:
                    keep_mask = 1  # - tokens_small_obj
                else:
                    keep_mask = 1 - tokens_small_obj
            else:
                keep_mask = 1 - tokens_to_discard
        elif self.token_split_conv.expand == 2:
            # TODO: check if logic_add is not differentiable?
            # old version for split token from c to 2C
            if self.training:
                keep_mask = 1 - tokens_to_discard  # + tokens_small_obj
            else:
                keep_mask = 1 - (tokens_to_discard + tokens_small_obj)
        else:
            raise NotImplementedError

        # x_new = x.clone() * keep_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) + x_small  #
        # keep_mask = 1 - (tokens_to_discard + tokens_small_obj)  # N, B,

        if final_split_num > 0:  # tokens_to_discard_original.sum() > 0 cannot catch the case

            if self.training:  # TODO: adding new is no good, we should use avg
                # 
                # x_new = (x.clone() * keep_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) + x_small) # t
                x_new = x.clone() * keep_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) * (
                        1 - x_weight) + x_small * x_weight

            else:
                x_new = x.clone() * keep_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) + x_small  # t
        else:  # no any token to split, to make it differentiable use significance_score * 0
            x_new = x.clone() + (significance_score * 0).unsqueeze(-1).repeat(1, 1, x.shape[-1]) + x_small

        # print(f'x_small.requires_grad = {x_small.requires_grad}')
        # print(f'x_new.requires_grad = {x_new.requires_grad}')
        invalid_tokens = ~get_valid_token_mask(x, mask)
        src_mask_reclaimed = None
        if reclaim_padded_region:
            src_mask_reclaimed = self.get_attn_mask_for_claimed_padded_regions(
                invalid_tokens=invalid_tokens, tokens_to_discard=tokens_to_discard_original)

        # We cannot mask all tokens as invalid.
        if src_mask_reclaimed is not None:
            for src_mask in src_mask_reclaimed:  # B, N
                assert not src_mask.all()

        output_dict = {
            'x': x_new,
            'src_mask_reclaimed': src_mask_reclaimed,
            'tokens_small_obj': tokens_small_obj.bool(),
            'tokens_to_discard': tokens_to_discard.bool(),
            'valid_tokens': valid_tokens_float,  # TODO: change this to bool()

            # valid_tokens in the original size, this is only used for loss calculation.
            'valid_tokens_original': valid_tokens_float,
            'increase_resolution': False,

            # for visualization only
            'tokens_to_discard_original': tokens_to_discard_original,
            'tokens_to_split_original': tokens_small_obj,
            # 'significance_score': small_scale_score,  # small_scale_score  fg_score

            'fg_score': significance_score,
            'small_scale_score': significance_score,
        }

        output_dict.update({'fg_score_logit': significance_score_logit, })

        if self.training:
            # This must be passed out for training.
            # TODO: handle the pos encoding update inside sgdt instead of Transformer.
            output_dict.update({'split_tok_indicators': split_tok_indicators})
        return output_dict

    # def discard_split_gumbel_softmax(
    #         self, x, mask=None,  #
    #         feat_map_size=None,
    #         with_global_feat=None,
    #         reclaim_padded_region=False,  # token split, remove
    #         no_bg_token_remove=False,
    #         no_token_split=False,
    #         **kwargs
    # ):
    #     """
    #     scale guided supervision differentiable, but sumbel_softmax also differentiable
    #     we need to decouple these two processes to check their effects.
    #     Using predicted labels as selection results is also a good choice. I need to test this as well.
    #     Args:
    #         no_bg_token_remove: for testing the idea of only using the padded regions
    #         reclaim_padded_region:
    #         x:  N, B, C = x.shape
    #         mask:
    #
    #     Returns:
    #
    #     """
    #     # (N, B, 1) -> (N, B) e.g., torch.Size([572, 2, 1])
    #     significance_score_logit = self.token_scoring(x).squeeze(-1)  #
    #     # -------------------------------------
    #     # significance_score = F.sigmoid(significance_score_logit).detach().clone()  # torch.Size([572, 2, 1])
    #     significance_score = F.sigmoid(significance_score_logit)  # torch.Size([572, 2, 1])
    #     # -------------------------------------
    #
    #     # token sampling gumbel_softmax
    #     valid_tokens_float = get_valid_token_mask(x, mask).float()
    #     fg_mask = (significance_score >= FG_SIGNIFICANCE_THD).float() * valid_tokens_float  # 0.5 or MIN_FG_SIGNIFICANCE
    #     # no need MAX_BG_SIGNIFICANCE, as the tokens used as bg is selected from highly confident ones.
    #     bg_mask = 1 - fg_mask
    #     # bg_mask = (significance_score < MAX_BG_SIGNIFICANCE).float()
    #
    #     invalid_tokens = ~get_valid_token_mask(x, mask)
    #
    #     # background tokens
    #     tokens_to_discard_original = bg_mask * valid_tokens_float  # (N, B)
    #     tokens_to_split_original = fg_mask
    #
    #     if reclaim_padded_region:
    #         if not no_bg_token_remove:  # predicted bg for valid tokens + padded region
    #             # TODO: check if detach is needed
    #             tokens_to_discard_original = (bg_mask + invalid_tokens).bool().float()  # .detach().clone()
    #
    #             # update significance_score so that tokens for padded regions has low scores (low priority of being
    #             # sampled. (later we can change this to high).
    #             significance_score[invalid_tokens] = float("-inf")  # -1e6  # set the score to be very lower
    #
    #         else:  # 2) padded region only
    #             tokens_to_discard_original = invalid_tokens.float()  # .detach().clone()
    #
    #     # assert not(torch.logical_and(tokens_to_discard_original, tokens_to_split_original).any()), \
    #     #     'There should be no overlap for tokens_to_discard_original and tokens_to_split'
    #
    #     assert (tokens_to_discard_original * tokens_to_split_original).sum() == 0, \
    #         'There should be no overlap for tokens_to_discard_original and tokens_to_split'
    #
    #     assert bg_mask[invalid_tokens].sum() == invalid_tokens.sum(), \
    #         'All invalid_tokens should be regarded as bg, no any invalid_tokens should be fg.'
    #
    #     x_new, tokens_to_discard, tokens_to_split = self._extract_adapted_token(
    #         x=x, tokens_to_discard_original=tokens_to_discard_original,
    #         tokens_to_split_original=tokens_to_split_original,
    #         fg_score=significance_score,
    #         small_scale_score=significance_score
    #     )
    #
    #     src_mask_reclaimed = None
    #     if reclaim_padded_region:
    #         src_mask_reclaimed = self.get_attn_mask_for_claimed_padded_regions(
    #             invalid_tokens=invalid_tokens, tokens_to_discard=tokens_to_discard)
    #
    #     token_dict = {
    #         'x': x_new,
    #         'src_mask_reclaimed': src_mask_reclaimed,
    #         'invalid_tokens': invalid_tokens,
    #         'tokens_small_obj': tokens_to_split,
    #         'tokens_to_discard': tokens_to_discard,
    #
    #         'fg_mask': fg_mask,
    #         'fg_obj_score': significance_score,
    #         'tokens_to_discard_original': tokens_to_discard_original,
    #
    #         'small_scale_score': significance_score,
    #         'scale_mask': fg_mask,
    #         'tokens_to_split_original': tokens_to_split_original,
    #     }
    #
    #     fg_score_logit = None
    #     scale_score_logit = significance_score_logit
    #     # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]), probability (sum to 1 for each prediction of one token)
    #     return token_dict, fg_score_logit, scale_score_logit, valid_tokens_float

    # def discard_split_gumbel_softmax1(self, x, mask=None, scale_fg_independent=False):
    #     """
    #     scale guided supervision differentiable, but sumbel_softmax also differentiable
    #     we need to decouple these two processes to check their effects.
    #     Using predicted labels as selection results is also a good choice. I need to test this as well.
    #     Args:
    #         scale_fg_independent:
    #         x:
    #         mask:
    #
    #     Returns:
    #
    #     """
    #     N, B, C = x.shape
    #     # token scoring, return N, B, Num_Classes (2 for fg, 3 for scale).
    #     fg_score_logit, scale_score_logit = self.token_scoring(x)
    #     # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]); torch.float32
    #
    #     lsx = nn.LogSoftmax(dim=-1)  # exp(lsx) can get the softmax score, here we use lsx for gumbel_softmax
    #     fg_score, scale_score = lsx(fg_score_logit), lsx(scale_score_logit)
    #
    #     # token sampling
    #     fg_mask = F.gumbel_softmax(fg_score, hard=True, dim=-1)  # 0 bg, 1 fg.
    #     # foreground score as significance_score
    #     significance_score = torch.exp(fg_score[:, :, -1])  # (N, B) torch.Size([650, 2])
    #
    #     #  0: non-small (medium or large); 1: small.
    #     #  (old definition 0, small, 1, medium, 2, large is deprecated)
    #     scale_mask = F.gumbel_softmax(scale_score, hard=True, dim=-1)  # TODO: tau=1 set temperature
    #     small_scale_score = torch.exp(scale_score[:, :, -1])  # TODO: test fg_score * small_score
    #
    #     # extract discard-tokens, small-object-tokens
    #     valid_tokens = get_valid_token_mask(x, mask).float()
    #     invalid_tokens = ~(get_valid_token_mask(x, mask))
    #     # to extract tokens inside image region (exclude padded regions)
    #     # tokens_to_discard = torch.logical_and(fg_mask[:, :, 0].bool(),
    #     #                                       valid_tokens)  # (N, B) num to discard: tensor([497., 363.]
    #
    #     # require_grad = True for tokens_to_discard_original and tokens_to_split_original
    #     tokens_to_discard_original_raw = fg_mask[:, :, 0] * valid_tokens  # (N, B) num to discard: tensor([497., 363.]
    #     # foreground and small objects, (N, B)
    #
    #     if scale_fg_independent:
    #         # keep scale_mask and fg_mask independent to simplify the training task.
    #         tokens_to_split_original_raw = scale_mask[:, :, 1] * valid_tokens
    #     else:
    #         tokens_to_split_original_raw = fg_mask[:, :, 1] * scale_mask[:, :, 1] * valid_tokens
    #
    #     # The tokens both to split or remove.
    #     ambiguous_tokens = tokens_to_discard_original_raw * tokens_to_split_original_raw
    #
    #     # Remove the ambiguous_tokens from the remove set and split set.
    #     tokens_to_discard_original = tokens_to_split_original_raw - ambiguous_tokens
    #     tokens_to_split_original = tokens_to_split_original_raw - ambiguous_tokens
    #
    #     tokens_to_discard = tokens_to_discard_original.clone().detach().bool()
    #     # foreground and small objects, (N, B)
    #     tokens_small_obj = tokens_to_split_original.clone().detach().bool()
    #     # calculate the number of tokens to split
    #     token_num = torch.stack([
    #         torch.sum(tokens_to_discard, dim=0),  # token to discard
    #         torch.sum(tokens_small_obj, dim=0),  # small object tokens
    #         # to control the maximum number of tokens to split
    #         torch.ones_like(torch.sum(tokens_to_discard, dim=0)) * self.max_split_token_num,
    #     ], dim=0)  # (3, B)
    #     #  (1, B), first column, the count for the first image; second column, second image.
    #     min_token_num = torch.min(token_num, dim=0)[0]  # 0, value; 1, indices
    #
    #     # sample discard tokens if there are more tokens_to_discard than tokens_small_obj
    #     batch_ids = (token_num[0] - min_token_num > 0).nonzero(as_tuple=True)[0]
    #     # if (token_num[0] > min_token_num).sum() > 0:
    #     for k in batch_ids:
    #         # significance_score[tokens_to_discard] will return a 1d vector and thus does not work
    #         # Returns the indices that sort a tensor along a given dimension
    #         # in ascending order (in default) by value.
    #         sorted_ids = torch.argsort(significance_score[tokens_to_discard[:, k], k], descending=True)  #
    #         ids = sorted_ids[min_token_num[k]:]
    #         # torch.nonzero return N,1 tensor
    #         tokens_to_stop_discard = torch.nonzero(tokens_to_discard[:, k]).squeeze(-1)[ids]
    #         # change its value from True to False to disable discarding
    #         tokens_to_discard[tokens_to_stop_discard, k] = False
    #
    #     # if (token_num[1] > min_token_num).sum() > 0:
    #     batch_ids = (token_num[1] - min_token_num > 0).nonzero(as_tuple=True)[0]
    #     for k in batch_ids:  # TODO: put this to a function
    #         sorted_ids = torch.argsort(small_scale_score[tokens_small_obj[:, k], k], descending=True)  #
    #         ids = sorted_ids[min_token_num[k]:]
    #
    #         tokens_to_stop_split = torch.nonzero(tokens_small_obj[:, k]).squeeze(-1)[ids]
    #         tokens_small_obj[tokens_to_stop_split, k] = False
    #
    #     x_small = torch.zeros_like(x)
    #     for k in range(B):
    #         # different image has different number of tokens to split or merge, thus
    #         # batch processing is not possible.
    #         img_small_obj_ids = tokens_small_obj[:, k]
    #         img_discard_token_ids = tokens_to_discard[:, k]
    #         x_k = x[:, k, :][img_small_obj_ids]  # M, C, where M is the number of tokens to split
    #         tokens_small_obj_new = self.token_split_conv(x_k)
    #         # # x[img_small_obj_ids, k, :] = tokens_small_obj_new[0]
    #         # # x[img_discard_token_ids, k, :] = tokens_small_obj_new[1]
    #
    #         x_small[img_small_obj_ids, k, :] += tokens_small_obj_new[0]
    #         x_small[img_discard_token_ids, k, :] += tokens_small_obj_new[1]
    #     # # TODO: directly modify the value, differentiable? to check, maybe it is differentiable
    #     keep_mask = 1 - (tokens_to_discard_original * tokens_to_discard + tokens_to_split_original * tokens_small_obj)
    #     x_new = x.clone() * keep_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) + x_small  # t
    #
    #     token_dict = {
    #         'x': x_new,
    #
    #         'src_mask_reclaimed': None,
    #         # 'attn_masks': attn_masks,
    #         # 'token_num': N,
    #         # 'map_size': [H, W],
    #         # 'init_grid_size': [H, W],
    #         # 'idx_token': idx_token,
    #         # 'agg_weight': agg_weight,
    #         'invalid_tokens': invalid_tokens,
    #
    #         'tokens_small_obj': tokens_small_obj,
    #         'tokens_to_discard': tokens_to_discard,
    #
    #         'fg_mask': fg_mask,
    #         'fg_obj_score': significance_score,
    #         'tokens_to_discard_original': tokens_to_discard_original,
    #
    #         'small_scale_score': small_scale_score,
    #         'scale_mask': scale_mask,
    #         'tokens_to_split_original': tokens_to_split_original,
    #     }
    #     # not significance_score, small_scale_score but all predictions, fg_score, torch.exp(scale_score)
    #     return token_dict, fg_score_logit, scale_score_logit, valid_tokens
    #     # torch.Size([650, 2, 2]), torch.Size([630, 2, 3]), probability (sum to 1 for each prediction of one token)

    def discard_split_gumbel_softmax(self, x, mask=None,
                                     feat_map_size=None,
                                     with_global_feat=None,
                                     reclaim_padded_region=False,  # token split, remove
                                     no_bg_token_remove=False,
                                     no_token_split=False,
                                     **kwargs):

        significance_score_logit = self.token_scoring(x, feat_map_size=feat_map_size,
                                                      with_global_feat=with_global_feat,
                                                      mask=mask)

        lsx = nn.LogSoftmax(dim=-1)  # exp(lsx) can get the softmax score, here we use lsx for gumbel_softmax
        scale_score = lsx(significance_score_logit)
        importance_mask = F.gumbel_softmax(scale_score, hard=True, dim=-1)  # TODO: tau=1 set temperature

        valid_tokens_float = get_valid_token_mask(x, mask).float()
        # invalid_tokens = ~(get_valid_token_mask(x, mask))

        tokens_to_discard_original = 1 - valid_tokens_float  # requires_grad = True
        tokens_to_split_original = importance_mask[:, :, 1] * valid_tokens_float  # 1 means the last dimension

        significance_score = torch.exp(scale_score[:, :, -1])  #
        significance_score_updated = significance_score * valid_tokens_float

        N, B, C = x.shape
        x_small = torch.zeros_like(x)

        tokens_to_split = torch.zeros_like(valid_tokens_float)
        tokens_to_discard = torch.zeros_like(valid_tokens_float)

        invalid_token_num = torch.sum((1 - valid_tokens_float), dim=0).int()  # (N, B)

        final_split_num = 0
        for k in range(B):
            num_split = tokens_to_split_original[:, k].sum().item()
            max_split_token_num = int(min(valid_tokens_float[:, k].sum().item(),
                                          invalid_token_num[k].item(),
                                          num_split,
                                          self.max_split_token_num))
            final_split_num += max_split_token_num
            # if max_split_token_num == 0:
            #     continue

            tokens_to_discard_inds = torch.nonzero(tokens_to_discard_original[:, k]).squeeze(-1)[:max_split_token_num]
            tokens_to_discard[tokens_to_discard_inds, k] = 1
            img_discard_token_ids = tokens_to_discard[:, k].bool()

            # Randomly select the valid split tokens, instead of selecting them from high score to low score, so
            # that we can enforce gradient on all valid tokens eventually.
            tokens_to_split_inds = torch.nonzero(tokens_to_split_original[:, k]).squeeze(-1)
            inds = torch.randperm(tokens_to_split_inds.shape[0])[:max_split_token_num].to(
                device=tokens_to_split_inds.device)
            tokens_to_split_inds = tokens_to_split_inds[inds]

            tokens_to_split[tokens_to_split_inds, k] = 1
            img_split_token_ids = tokens_to_split[:, k].bool()

            x_k = x[:, k, :][img_split_token_ids]  # M, C, where M is the number of tokens to split
            tokens_small_obj_new = self.token_split_conv(x_k)

            x_small[img_split_token_ids, k, :] = tokens_small_obj_new[0]
            # x_new[img_small_obj_ids, k, :] = x_new[img_small_obj_ids, k, :] * 0 + tokens_small_obj_new[0]
            # '+' to make it differentiable
            x_small[img_discard_token_ids, k, :] = tokens_small_obj_new[1]

        # # there should be no token in both tokens_small_obj and tokens_to_discard
        assert (tokens_to_discard * tokens_to_split).max() < 1

        # tokens_to_split_original * tokens_to_split is to make significance_score_logit differential
        keep_mask = 1 - (tokens_to_discard_original * tokens_to_discard +
                         tokens_to_split_original * tokens_to_split)  # N, B,

        # if final_split_num > 0:  # tokens_to_discard_original.sum() > 0 cannot catch the case
        x_new = x.clone() * keep_mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) + x_small  # t
        # else:  # TODO: no need this.
        #     x_new = x.clone() + (significance_score * 0).unsqueeze(-1).repeat(1, 1, x.shape[-1]) + x_small  # t

        # print(f'x_small.requires_grad = {x_small.requires_grad}')
        # print(f'x_new.requires_grad = {x_new.requires_grad}')
        invalid_tokens = ~get_valid_token_mask(x, mask)
        src_mask_reclaimed = None
        if reclaim_padded_region:
            src_mask_reclaimed = self.get_attn_mask_for_claimed_padded_regions(
                invalid_tokens=invalid_tokens, tokens_to_discard=tokens_to_discard_original)

        # We cannot mask all tokens as invalid.
        if src_mask_reclaimed is not None:
            for src_mask in src_mask_reclaimed:  # B, N
                assert not src_mask.all()

        output_dict = {
            'x': x_new,
            'src_mask_reclaimed': src_mask_reclaimed,
            'tokens_small_obj': tokens_to_split.bool(),
            'tokens_to_discard': tokens_to_discard.bool(),
            'valid_tokens': valid_tokens_float,  # TODO: change this to bool()

            # valid_tokens in the original size, this is only used for loss calculation.
            'valid_tokens_original': valid_tokens_float,
            'increase_resolution': False,

            # for visualization only
            'tokens_to_discard_original': tokens_to_discard_original,
            'tokens_to_split_original': tokens_to_split,
            # 'significance_score': small_scale_score,  # small_scale_score  fg_score

            'fg_score': significance_score_updated,
            'small_scale_score': significance_score_updated,

        }

        output_dict.update(
            {'fg_score_logit': significance_score_logit}
        )
        return output_dict

    @staticmethod
    def get_attn_mask_for_masking_out_removed_tokens(invalid_tokens, removed_tokens_mask):
        """ When only removing tokens, no splitting tokens, we need to generate the attn masks.

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
            assert not src_mask.all()
            # if src_mask.all():
            #     inds = removed_tokens_mask[:, k].nonzero(as_tuple=True)[0]
            #     # set the first item to be False, to set this location to be valid.
            #     src_mask_reclaimed[k, inds[0]] = False
        return src_mask_reclaimed

    @staticmethod
    def get_attn_mask_for_claimed_padded_regions(invalid_tokens, tokens_to_discard):
        """ Note: this version is only for adapted tokens as query, original token as key,
        If this is not the case, then we need to set src_mask_reclaimed = None in EncoderLayer (not here).

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

    @staticmethod
    def get_attn_mask_for_claimed_padded_regions_debug(invalid_tokens, tokens_to_discard, tokens_to_split):
        """ Note: this version is only for adapted tokens as query, original token as key,
        If this is not the case,

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

        # set the tokens to split as invalid, only for debugging purpose.
        src_mask_reclaimed[tokens_to_split] = True

        src_mask_reclaimed = src_mask_reclaimed.permute(1, 0)  # N, B ->  B, N

        return src_mask_reclaimed

    def forward(self, x, mask=None,
                # reclaim_padded_region=False,
                sgdt_targets=None,  # only for debugging
                feat_map_size=None,
                sigma=None,
                gt_ratio=None,
                ):
        """

        Args:
            x:
            mask:

        Returns:

        """

        if not self.training:
            self.max_split_token_num = self.max_split_token_num_inference

        # return self.discard_split(x, mask)

        reclaim_padded_region = self.token_scoring_config_parser.reclaim_padded_region
        no_bg_token_remove = self.token_scoring_config_parser.no_bg_token_remove
        no_token_split = self.token_scoring_config_parser.no_split

        # return self.discard_split_with_gt_token_score(x, mask, sgdt_targets=sgdt_targets)
        if self.token_scoring_discard_split_criterion.find('gt_only_exp') > -1:
            if self.token_scoring_config_parser.str_exist('debug_split_1c'):
                return self.discard_split_with_gt_token_score_debug_split_1c(
                    x, mask, reclaim_padded_region=reclaim_padded_region,
                    sgdt_targets=sgdt_targets,
                    no_bg_token_remove=no_bg_token_remove,
                    no_token_split=no_token_split,
                    # debug_gt_split_ratio=debug_gt_split_ratio,
                    # debug_gt_remove_ratio=debug_gt_remove_ratio,
                    #
                    # proposal_split_thd=proposal_split_thd,
                    # proposal_remove_thd=proposal_remove_thd,

                )
            else:
                return self.discard_split_with_gt_token_score(
                    x, mask, reclaim_padded_region=reclaim_padded_region,
                    sgdt_targets=sgdt_targets,
                    no_bg_token_remove=no_bg_token_remove,
                    no_token_split=no_token_split,
                    gt_ratio=gt_ratio,
                    # debug_gt_split_ratio=debug_gt_split_ratio,
                    # debug_gt_remove_ratio=debug_gt_remove_ratio,
                    #
                    # proposal_split_thd=proposal_split_thd,
                    # proposal_remove_thd=proposal_remove_thd,

                )
        else:
            with_global_feat = self.token_scoring_config_parser.with_global_feat

            inverse_split_thd = self.token_scoring_config_parser.inverse_split_thd
            filter_false_split = self.token_scoring_config_parser.filter_false_split
            split_sig_thd = self.token_scoring_config_parser.split_sig_thd
            inverse_bg_thd = self.token_scoring_config_parser.inverse_remove_thd
            bg_sig_thd = self.token_scoring_config_parser.bg_sig_thd
            filter_false_remove = self.token_scoring_config_parser.filter_false_remove

            # predict scale directly
            if self.token_scoring_config_parser.predict_significance:
                return self.discard_split_significance_all_fg_w_priority(
                    x, mask=mask,
                    with_global_feat=with_global_feat,
                    feat_map_size=feat_map_size,
                    reclaim_padded_region=reclaim_padded_region,
                    no_bg_token_remove=no_bg_token_remove,
                    no_token_split=no_token_split,
                    # the following parameters are for debugging
                    sgdt_targets=sgdt_targets,

                    split_sig_thd=split_sig_thd,
                    inverse_split_thd=inverse_split_thd,
                    filter_false_split=filter_false_split,
                    bg_sig_thd=bg_sig_thd,
                    inverse_bg_thd=inverse_bg_thd,
                    filter_false_remove=filter_false_remove
                )
            elif self.token_scoring_config_parser.str_exist('topk_token_selection_differentiable'):
                return self.discard_split_topk_token_selection_differentiable(
                    x, mask=mask,
                    sgdt_targets=sgdt_targets,
                    with_global_feat=with_global_feat,
                    feat_map_size=feat_map_size,
                    reclaim_padded_region=reclaim_padded_region,
                    no_bg_token_remove=no_bg_token_remove,
                    no_token_split=no_token_split,

                    # TODO: check if we need the following split setting.
                    split_sig_thd=split_sig_thd,
                    inverse_split_thd=inverse_split_thd,
                    filter_false_split=filter_false_split,

                    bg_sig_thd=bg_sig_thd,
                    inverse_bg_thd=inverse_bg_thd,
                    filter_false_remove=filter_false_remove,
                    sigma=sigma
                )
            elif self.token_scoring_config_parser.str_exist('gumbel_softmax_token_selection_differentiable'):
                return self.discard_split_gumbel_softmax(
                    x, mask=mask,
                    with_global_feat=with_global_feat,
                    feat_map_size=feat_map_size,
                    reclaim_padded_region=reclaim_padded_region,
                    no_bg_token_remove=no_bg_token_remove,
                    no_token_split=no_token_split,

                    sgdt_targets=sgdt_targets,
                    # TODO: check if we need the following split setting.
                    split_sig_thd=split_sig_thd,
                    inverse_split_thd=inverse_split_thd,
                    filter_false_split=filter_false_split,

                    bg_sig_thd=bg_sig_thd,
                    inverse_bg_thd=inverse_bg_thd,
                    filter_false_remove=filter_false_remove,
                    # sigma=sigma
                )

            elif self.token_scoring_config_parser.str_exist('pred_token_fg'):

                # inverse_bg_thd = self.token_scoring_config_parser.inverse_remove_thd
                # bg_sig_thd = self.token_scoring_config_parser.bg_sig_thd
                # filter_false_remove = self.token_scoring_config_parser.filter_false_remove

                return self.discard_split_pred_fg(
                    x, mask=mask,
                    with_global_feat=with_global_feat,
                    feat_map_size=feat_map_size,
                    reclaim_padded_region=reclaim_padded_region,
                    no_bg_token_remove=no_bg_token_remove,
                    no_token_split=no_token_split,

                    sgdt_targets=sgdt_targets,
                    # TODO: check if we need the following split setting.
                    split_sig_thd=split_sig_thd,
                    inverse_split_thd=inverse_split_thd,
                    filter_false_split=filter_false_split,

                    bg_sig_thd=bg_sig_thd,
                    inverse_bg_thd=inverse_bg_thd,
                    filter_false_remove=filter_false_remove
                )

            # ------------ not cleaned yet.
            elif self.token_scoring_discard_split_criterion == 'test_but_scoring_grad_for_loss_only':
                return self.discard_split_with_gt_token_score_but_pred_scoring(
                    x, mask, reclaim_padded_region=False,
                    sgdt_targets=sgdt_targets,
                    no_bg_token_remove=False
                )
            else:
                raise NotImplementedError


