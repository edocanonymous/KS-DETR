import torch
import torch.nn.functional as F
from torch import nn


# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from models.label_smooth.label_smoothing import LabelSmoothingCrossEntropy

# https://github.com/huawei-noah/Pretrained-Language-Model/blob/ad3a1960780340e1a8fb1136b25ff14811ae5cf5/TinyBERT/general_distill.py#L430


class TokenPredLoss:

    @torch.no_grad()
    def _cal_accu(self, pred_logits, targets, valid_token_mask, cal_per_img_accu=False):
        """
        Args:
            pred_logits: N, B, Num_class
            targets: N, B
            valid_token_mask: N, B (bool)

        Returns:

        """
        if cal_per_img_accu:
            accu_list = []
            N, B = targets.shape
            for k in range(B):
                pred = pred_logits[:, k, :].max(dim=-1).indices[valid_token_mask[:, k]]
                tgt = targets[:, k][valid_token_mask[:, k]]
                accu = self.estimate_accu(pred, tgt)
                accu_list.append(accu)
            return accu_list
        else:
            if pred_logits.ndim == 3:  # or pred_logits.dim()
                pred_logits, targets = self.reshape_logits_targets(pred_logits, targets)

            preds = pred_logits.max(dim=-1).indices
            assert preds.shape == valid_token_mask.shape == targets.shape

            accu = self.estimate_accu(
                preds[valid_token_mask],  # .reshape(-1)
                targets[valid_token_mask])
        return accu

    def _cal_ce_loss(self, pred_logits, targets, valid_tokens, soft_label=False,
                     pos_weight_ratio=1.0,
                     fg_or_split_token_labels=None
                     ):
        """

        Args:
            pred_logits: (N, B, Num_class)
            targets: (N, B)
            valid_tokens: (N, B)
            soft_label:
            pos_weight_ratio: float

        Returns:

        """
        assert isinstance(pos_weight_ratio, (int, float))
        num_class = pred_logits.shape[-1]
        assert num_class == 2, 'Currently only support binary classification problem'

        assert not soft_label, 'Currently not support soft label'

        device = pred_logits.device
        weights = torch.tensor([1.0, pos_weight_ratio]).to(device)
        loss_criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=weights)  #

        # pred_logits: (N, B, Num_class) -> (N * B, Num_class);
        # targets: (N, B) -> (N * B)
        pred_logits, targets = self.reshape_logits_targets(pred_logits, targets)

        # if torch.logical_and((targets > 0), targets < 1).float().sum() > 0:
        if soft_label:
            targets_new = targets.unsqueeze(-1)
            targets_new = torch.cat([1 - targets_new, targets_new], dim=-1)
        else:
            assert torch.logical_and((targets > 0), targets < 1).float().sum() == 0, 'Only 0 and 1 values' \
                                                                                     'are allowed in targets.'
            targets_new = targets.long()

        ce_loss = loss_criterion(pred_logits, targets_new)  # torch.Size([713, 2])

        # sample_weights = torch.full_like(targets, 1.0)
        # sample_weights[targets > 0] = pos_weight_ratio
        valid_tokens_float = valid_tokens.float().reshape(-1)
        final_loss = (ce_loss * valid_tokens_float).sum() / valid_tokens_float.sum()
        # final_loss = (ce_loss * sample_weights * valid_tokens_float).sum() / \
        #              (sample_weights * valid_tokens_float).sum()

        # calculate accuracy
        valid_token_mask = valid_tokens_float.bool()
        pred_labels = pred_logits.max(dim=-1).indices.reshape(-1)
        accu = self.estimate_accu(pred_labels[valid_token_mask], targets[valid_token_mask])

        # num of fg being misclassified as bg
        num_fg_tokens = targets.sum()
        num_false_bg = (1 - pred_labels[targets.bool()]).sum()

        num_false_bg_custom_thd = float('inf')
        if fg_or_split_token_labels is not None:
            num_false_bg_custom_thd = (1 - fg_or_split_token_labels.view(-1)[targets.bool()]).sum()
            # if num_false_bg_custom_thd > 0:
            #     print(f'num_false_bg_custom_thd = {num_false_bg_custom_thd}')

        return final_loss, accu, num_false_bg, num_fg_tokens, num_false_bg_custom_thd

    @staticmethod
    def estimate_accu(pred, y):
        acc = (pred == y).sum() / pred.size(0)  # keep it as tensor, so not use  .item()
        return acc

    @staticmethod
    def reshape_logits_targets(pred_logits, targets):
        # merge N, B into a single dimension (N * B)

        N, B, Num_Class = pred_logits.shape
        pred_logits = pred_logits.reshape(N * B, -1)  # N, B, Num_Class ->  N x B, Num_Class
        # if targets.ndim == 3:  # for soft label task
        #     targets = targets.reshape(N * B, -1)
        # else:
        #     targets = targets.reshape(N * B)  # N, B -> N x B, 2
        targets = targets.reshape(N * B)  # N, B -> N x B
        return pred_logits, targets

    def cal_focal_loss_and_accu(self, pred_logits, token_targets, valid_tokens, pos_mask=None,
                                cal_per_img_accu=False,
                                alpha: float = 0.25, gamma: float = 2):
        """

        Args:
            pred_logits: N, B, C (number of class)
            token_targets: N, B
            valid_tokens: N, B
            pos_mask:  N, B
            alpha:
            gamma:

        Returns:

        """
        loss_criterion = torch.nn.CrossEntropyLoss(reduction='none')  # =0.1 epsilon=0,
        # loss_criterion = LabelSmoothingCrossEntropy(epsilon=0.1, reduction='none')  # =0.1
        # N, B
        softmax = nn.Softmax(dim=-1)

        # prob = inputs.sigmoid()
        # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        assert torch.logical_and((token_targets > 0), token_targets < 1).float().sum() == 0, \
            'Only 0 and 1 values are allowed in targets.'

        accu = self._cal_accu(pred_logits, targets=token_targets,
                              valid_token_mask=valid_tokens, cal_per_img_accu=cal_per_img_accu)

        pred_logits, token_targets = self.reshape_logits_targets(pred_logits, token_targets)

        # prob = softmax(pred_logits)[:, :, -1]
        prob = softmax(pred_logits)[:, -1]
        ce_loss = loss_criterion(pred_logits, token_targets.long())  # torch.Size([713, 2])
        p_t = prob * token_targets + (1 - prob) * (1 - token_targets)
        loss_ = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * token_targets + (1 - alpha) * (1 - token_targets)
            loss_ = alpha_t * loss_
        # ignore the predictions for invalid tokens. valid_tokens (N, B)

        if pos_mask is not None:
            # only calculate the locations of fg tokens for small scale token prediction.
            mask_float = valid_tokens.float().reshape(-1) * pos_mask.reshape(-1).float()

        else:
            # loss_ = (loss_ * valid_tokens.reshape(-1)).sum() / valid_tokens.sum()  # B
            mask_float = valid_tokens.float().reshape(-1)

        loss_ = (loss_ * mask_float).sum() / mask_float.sum()  # B

        # valid_token_mask = mask_float.bool()
        # accu_again = self.estimate_accu(
        #     pred_logits.max(dim=-1).indices.reshape(-1)[valid_token_mask],
        #     token_targets.reshape(-1)[valid_token_mask])

        loss = dict(
            ksgt_loss_fg=loss_,  #
            ksgt_fg_bg_accu=accu,  # pred_logits.new_tensor(accu)
        )
        return loss


class TokenScoringLoss(TokenPredLoss):
    def __init__(self, token_scoring_loss_criterion):

        if token_scoring_loss_criterion == 'reg_sigmoid_l1_loss':
            self.loss_func = self.loss_scale_score_reg_sigmoid_l1
        elif token_scoring_loss_criterion == 'gt_fg_scale_fake':
            self.loss_func = self.loss_fg_scale_fake_for_gt
        elif token_scoring_loss_criterion == 'fg_scale_class_pred_focal_loss':
            self.loss_func = self.focal_loss_fg_scale_score
        elif token_scoring_loss_criterion == 'fg_scale_class_pred_ce_independent':
            self.loss_func = self.loss_fg_scale_score_ce
        elif token_scoring_loss_criterion == 'fg_weighted_ce':
            self.loss_func = self.loss_fg_weighted_ce
        else:
            raise NotImplementedError

    def cal_loss(self, ksgt_output, ksgt_targets):
        return self.loss_func(ksgt_output, ksgt_targets)

    def loss_scale_score_reg_sigmoid_l1(self, ksgt_output, ksgt_targets, over_weigh_fg_loss=False,
                                        # pos_weight_ratio=1.0
                                        ):
        """
        loss for a single encoder ksgt layer for scoring supervision of significance prediction.
        Args:

            over_weigh_fg_loss: True is no good, cause all prediction to be 1.0 (small object).
            ksgt_output: a dict,
                     dict(
                        small_scale_score_logit=small_scale_score_logit,  # # B, N; torch.Size([2, 630])
                    )
            ksgt_targets: a dict,
                dict(
                        scale_gt, float(), [0, 1]  #  N, B
                )
        )
        Returns:

        """

        valid_tokens_float = ksgt_output['valid_tokens']  # valid_tokens:  (N, B), 1 valid, 0, invalid
        valid_token_mask = ksgt_output['valid_tokens'].bool().reshape(-1)

        # logits to range [0, 1]
        pred_logits = torch.sigmoid(ksgt_output['small_scale_score_logit'])  # N, B
        targets = ksgt_targets['scale_gt']  # N, B
        pred_flattened, targets_flattened = pred_logits.reshape(-1), targets.reshape(-1)

        loss_token_significance = F.l1_loss(pred_flattened, targets_flattened, reduction='none')
        # invalid tokens do not contribute to the final loss.
        weights = torch.ones_like(targets_flattened) * valid_tokens_float.reshape(-1)
        if over_weigh_fg_loss:
            # more significant locations have large weights, 0 -> 1, 0.6 -> 6.0; 1 -> 10.0
            weights[targets_flattened > 0] = weights[targets_flattened > 0] * 10

            # ignore the predictions for invalid tokens. valid_tokens (N, B)
        loss_token_sig = (loss_token_significance * weights).sum() / weights.sum()  # B

        device = pred_logits.device
        with torch.no_grad():
            fg_pred = pred_flattened >= 0.5  # range [0, 1]
            fg_target = targets_flattened >= 0.5  # bg 0, fg [0.5, 1]
            fg_bg_accu = self.estimate_accu(fg_pred[valid_token_mask], fg_target[valid_token_mask])

            N, B = valid_tokens_float.shape
            object_scale_valid_loc = torch.logical_and(fg_target, valid_token_mask).split(N)

            small_scale_order_errors = 0
            # num_items = 0
            for k, valid_loc_k in enumerate(object_scale_valid_loc):
                object_scale_pred_ind = torch.argsort(pred_logits[:, k][valid_loc_k], descending=True)
                object_scale_target_ind = torch.argsort(targets[:, k][valid_loc_k], descending=True)

                num_item = object_scale_pred_ind.reshape(-1).shape[0]
                src = torch.range(0, num_item).to(device=device)
                pred_order = torch.zeros_like(pred_logits[:, k]).scatter_(
                    0, index=object_scale_pred_ind, src=src)
                target_order = torch.zeros_like(targets[:, k]).scatter_(
                    0, index=object_scale_target_ind, src=src)

                # inds = torch.tensor([3, 1])
                # a = torch.range(0, inds.reshape(-1).shape[0]) + 10
                # pred_order = torch.zeros(8)
                # result1 = pred_order.scatter_(0, inds, a)
                # # tensor([ 0., 11.,  0., 10.,  0.,  0.,  0.,  0.])

                small_scale_order_errors += torch.abs(pred_order - target_order).sum()

                # num_items += num_item
            small_scale_order_errors = small_scale_order_errors / B
        # down weight the loss of fg_loss, small_scale_loss
        loss = dict(
            # ksgt_loss_fg=fg_loss * self.ksgt_loss_weight,  # ,
            # ksgt_loss_small_scale=small_scale_loss * self.ksgt_loss_weight,  # ,
            ksgt_loss_token_significance=loss_token_sig,
            ksgt_fg_bg_accu=pred_logits.new_tensor(fg_bg_accu),  # pred_logits.new_tensor(fg_bg_accu)
            ksgt_small_scale_order_error=small_scale_order_errors,
        )
        token_update = self._extract_token_update_info(ksgt_output)
        loss.update(token_update)
        return loss

    def loss_fg_weighted_ce(self, ksgt_output, ksgt_targets):
        fg_loss, fg_accu, num_false_bg, num_fg_tokens, num_false_bg_custom_thd = self._cal_ce_loss(
            pred_logits=ksgt_output['fg_score_logit'],
            targets=ksgt_targets['fg_gt'],
            valid_tokens=ksgt_output['valid_tokens'].float(),
            soft_label=False,
            pos_weight_ratio=10,
            fg_or_split_token_labels=1 - ksgt_output['tokens_to_discard'],  # transfer to fg labels
        )

        # down weight the loss of fg_loss, small_scale_loss
        loss = dict(
            ksgt_loss_fg=fg_loss,  # ,  * self.ksgt_loss_weight
            ksgt_fg_accu=fg_loss.new_tensor(fg_accu),
            num_fg_tokens=num_fg_tokens,
            num_false_bg=num_false_bg,
            num_false_bg_custom_thd=num_false_bg_custom_thd,
            # ksgt_loss_small_scale=small_scale_loss,  # ,  * self.ksgt_loss_weight
            # ksgt_small_scale_accu=small_scale_loss.new_tensor(small_scale_accu),
            ksgt_num_tokens_to_split_remove=ksgt_output['tokens_to_discard'].sum(dim=0).float().mean(),
            ksgt_num_tokens_to_discard_original=ksgt_output['tokens_to_discard_original'].sum(dim=0).float().mean(),
            ksgt_num_tokens_to_split_original=ksgt_output['tokens_to_split_original'].sum(dim=0).float().mean(),
        )
        return loss

    @staticmethod
    def _extract_token_update_info(ksgt_output):
        out_dict = dict(
            ksgt_num_tokens_to_remove=ksgt_output['tokens_to_discard'].sum(dim=0).float().mean(),
            # ksgt_num_tokens_to_split=ksgt_output['tokens_to_split'].sum(dim=0).float().mean(),
            ksgt_num_tokens_to_discard_original=ksgt_output['tokens_to_discard_original'].sum(dim=0).float().mean(),
            ksgt_num_tokens_to_split_original=ksgt_output['tokens_to_split_original'].sum(dim=0).float().mean(),
        )
        return out_dict

    def loss_fg_scale_fake_for_gt(self, ksgt_output, ksgt_targets):
        """
        Does not make sense to calculate the accuracy since we use the gt.
        Args:
            ksgt_output:
            ksgt_targets:

        Returns:

        """
        # valid_token_mask = ksgt_output['valid_tokens'].bool().reshape(-1)
        #
        # fg_accu = self._cal_accu(
        #     pred_logits=ksgt_output['fg_score_logit'],  # torch.Size([775, 4, 2])
        #     targets=ksgt_targets['fg_gt'],  # torch.Size([775, 4])
        #     valid_token_mask=valid_token_mask
        # )
        # small_scale_accu = self._cal_accu(
        #     pred_logits=ksgt_output['small_scale_score_logit'],
        #     targets=ksgt_targets['scale_gt'].bool(),  # float to bool
        #     valid_token_mask=valid_token_mask
        # )
        # loss = dict(
        #     ksgt_fg_accu=fg_accu,
        #     ksgt_small_scale_accu=small_scale_accu,
        # )
        loss = self._extract_token_update_info(ksgt_output)
        # loss.update(token_update)
        return loss

    # TODO: need to be adapted.
    def focal_loss_fg_scale_score(self, ksgt_output, ksgt_targets,
                                  alpha: float = 0.25, gamma: float = 2,
                                  scale_fg_independent=True
                                  ):
        """
        loss for a single encoder ksgt layer for scoring supervision of fg and scale prediction.
        Args:
            # fg_score, scale_score  # N, B, C, where C is the number of classes,
            # e.g., torch.Size([650, 2, 2]), torch.Size([630, 2, 3]);
            each score: probability not logits (sum to 1 for each prediction of one token)

            ksgt_output: a dict,
                     dict(
                        adapted_pos=adapted_pos,
                        fg_score_logit=fg_score_logit,  # B, N; torch.Size([2, 630])
                        small_scale_score_logit=small_scale_score_logit,
                    )

            ksgt_targets: a dict,
                ksgt_targets = dict(  # B, H, W -> HxW, B, and then will be expanded to (N, B, Num_Class)
                        fg_gt=fg_gt.flatten(1).permute(1, 0), # N, B
                        scale_gt=scale_gt.flatten(1).permute(1, 0)   #  N, B
                )

        )
        Returns:

        """
        loss = {}

        loss_criterion = torch.nn.CrossEntropyLoss(reduction='none')  # =0.1 epsilon=0,
        # loss_criterion = LabelSmoothingCrossEntropy(epsilon=0.1, reduction='none')  # =0.1
        # N, B

        if 'increase_resolution' in ksgt_output and ksgt_output['increase_resolution']:
            valid_tokens = ksgt_output['valid_tokens_original'].float()
        else:
            valid_tokens = ksgt_output['valid_tokens'].float()  # valid_tokens:  (N, B), 1 valid, 0, invalid

        softmax = nn.Softmax(dim=-1)

        def cal_loss_accu(pred_logits, targets, pos_mask=None):
            # prob = inputs.sigmoid()
            # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            assert torch.logical_and((targets > 0), targets < 1).float().sum() == 0, 'Only 0 and 1 values' \
                                                                                     'are allowed in targets.'
            pred_logits, targets = self.reshape_logits_targets(pred_logits, targets)

            # prob = softmax(pred_logits)[:, :, -1]
            prob = softmax(pred_logits)[:, -1]
            ce_loss = loss_criterion(pred_logits, targets.long())  # torch.Size([713, 2])
            p_t = prob * targets + (1 - prob) * (1 - targets)
            loss_ = ce_loss * ((1 - p_t) ** gamma)

            if alpha >= 0:
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
                loss_ = alpha_t * loss_
            # ignore the predictions for invalid tokens. valid_tokens (N, B)

            if pos_mask is not None:
                # only calculate the locations of fg tokens for small scale token prediction.
                mask_float = valid_tokens.reshape(-1) * pos_mask.reshape(-1).float()

            else:
                # loss_ = (loss_ * valid_tokens.reshape(-1)).sum() / valid_tokens.sum()  # B
                mask_float = valid_tokens.reshape(-1)

            valid_token_mask = mask_float.bool()

            loss_ = (loss_ * mask_float).sum() / mask_float.sum()  # B
            accu = self.estimate_accu(
                pred_logits.max(dim=-1).indices.reshape(-1)[valid_token_mask],
                targets.reshape(-1)[valid_token_mask])
            return loss_, accu

        if 'fg_score_logit' in ksgt_output:
            fg_loss, fg_accu = cal_loss_accu(
                pred_logits=ksgt_output['fg_score_logit'],  # torch.Size([775, 4, 2])
                targets=ksgt_targets['fg_gt']  # torch.Size([775, 4])
            )
            loss.update(
                dict(
                    ksgt_loss_fg=fg_loss,  # ,  * self.ksgt_loss_weight
                    ksgt_fg_accu=fg_loss.new_tensor(fg_accu),
                )
            )
        if 'small_scale_score_logit' in ksgt_output:
            if scale_fg_independent:
                small_scale_loss, small_scale_accu = cal_loss_accu(
                    pred_logits=ksgt_output['small_scale_score_logit'],
                    targets=ksgt_targets['scale_gt']
                )
            else:
                small_scale_loss, small_scale_accu = cal_loss_accu(
                    pred_logits=ksgt_output['small_scale_score_logit'],
                    targets=ksgt_targets['scale_gt'],
                    pos_mask=ksgt_targets['fg_gt']
                )
            loss.update(
                dict(
                    ksgt_loss_small_scale=small_scale_loss,
                    ksgt_small_scale_accu=small_scale_loss.new_tensor(small_scale_accu)
                )
            )
        token_update = self._extract_token_update_info(ksgt_output)
        loss.update(token_update)
        return loss

    def loss_fg_scale_score_ce(self, ksgt_output, ksgt_targets):
        """
        loss for a single encoder ksgt layer for scoring supervision of fg and scale prediction.
        Args:
            # fg_score, scale_score  # N, B, C, where C is the number of classes,
            # e.g., torch.Size([650, 2, 2]), torch.Size([630, 2, 3]);
            each score: probability not logits (sum to 1 for each prediction of one token)

            ksgt_output: a dict,
                     dict(
                        adapted_pos=adapted_pos,
                        fg_score_logit=fg_score_logit,  # B, N; torch.Size([2, 630])
                        small_scale_score_logit=small_scale_score_logit,
                    )

            ksgt_targets: a dict,
                ksgt_targets = dict(  # B, H, W -> HxW, B, and then will be expanded to (N, B, Num_Class)
                        fg_gt=fg_gt.flatten(1).permute(1, 0), # N, B
                        scale_gt=scale_gt.flatten(1).permute(1, 0)   #  N, B
                )

        )
        Returns:

        """
        loss = {}

        if 'fg_score_logit' in ksgt_output:
            fg_loss, fg_accu = self._cal_ce_loss(
                pred_logits=ksgt_output['fg_score_logit'],
                targets=ksgt_targets['fg_gt'],
                valid_tokens=ksgt_output['valid_tokens'].float(),
                soft_label=False
            )
            loss.update(
                dict(
                    ksgt_loss_fg=fg_loss,  # ,  * self.ksgt_loss_weight
                    ksgt_fg_accu=fg_loss.new_tensor(fg_accu),
                )
            )

        if 'small_scale_score_logit' in ksgt_output:
            small_scale_loss, small_scale_accu = self._cal_ce_loss(
                pred_logits=ksgt_output['small_scale_score_logit'],
                targets=ksgt_targets['scale_gt'],
                valid_tokens=ksgt_output['valid_tokens'].float(),
                soft_label=True
            )
            loss.update(
                dict(
                    ksgt_loss_small_scale=small_scale_loss,
                    ksgt_small_scale_accu=small_scale_loss.new_tensor(small_scale_accu)
                )
            )
        token_update = self._extract_token_update_info(ksgt_output)
        loss.update(token_update)
        return loss

# class TokenScoringLossDeprecated(TokenScoringLoss):
#     pass
