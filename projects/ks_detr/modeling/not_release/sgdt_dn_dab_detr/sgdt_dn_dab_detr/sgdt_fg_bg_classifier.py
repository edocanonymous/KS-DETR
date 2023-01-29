import torch
from torch import nn


# ==================================== modification
from models.sgdt.sgdt_components import get_num_of_layer, src_key_padding_mask2valid_token_mask
from models.sgdt.sgdt_ import build_sgdt, FgBgClassifier
from models.sgdt.scoring_loss import TokenPredLoss
from models.sgdt.scoring_visualize import VisualizeToken

import numpy as np


class TokenPredStateV0:
    def __init__(self):
        self.accu_history = {}

    def update(self, image_id, accu):
        self.accu_history[image_id.cpu().item()] = accu.cpu().item()

    def batch_update(self, image_ids, accu_list):
        for img_id, acc in zip(image_ids, accu_list):
            self.update(image_id=img_id, accu=acc)

    @property
    def mean_accu(self):
        if len(self.accu_history) == 0:
            return 0.0

        return np.sum([x for x in self.accu_history.values()]) / len(self.accu_history)


class TokenPredStateV1:
    def __init__(self):
        self.accu_history = {}

    def update(self, layer_id, image_id, accu):
        if layer_id not in self.accu_history:
            self.accu_history[layer_id] = dict()

        self.accu_history[layer_id][image_id.cpu().item()] = accu.cpu().item()

    def batch_update(self, layer_id, image_ids, accu_list):
        for img_id, acc in zip(image_ids, accu_list):
            self.update(layer_id=layer_id, image_id=img_id, accu=acc)

    @property
    def mean_accu(self):
        if len(self.accu_history) == 0:
            return 0.0

        per_layer_mean_accu = []
        for layer_id, _ in self.accu_history.items():
            accu = self.layer_mean_accu(layer_id)
            per_layer_mean_accu.append(accu)
        return per_layer_mean_accu

    def layer_mean_accu(self, layer_id):
        accu_dict = self.accu_history[layer_id]
        # return torch.from_numpy((np.sum([x for x in accu_dict.values()]) / len(accu_dict))
        return np.sum([x for x in accu_dict.values()]) / len(accu_dict)


class TokenPredEvalState:
    def __init__(self):
        self.accu_history = {}

    def update(self, layer_id, image_id, accu):
        if layer_id not in self.accu_history:
            self.accu_history[layer_id] = dict()

        self.accu_history[layer_id][image_id.cpu().item()] = accu  # .cpu().item()

    def batch_update(self, layer_id, image_ids, accu_list):
        for img_id, acc in zip(image_ids, accu_list):
            self.update(layer_id=layer_id, image_id=img_id, accu=acc)

    @property
    def mean_accu(self):
        if len(self.accu_history) == 0:
            return 0.0

        per_layer_mean_accu = []
        for layer_id, _ in self.accu_history.items():
            accu = self.layer_mean_accu(layer_id)
            per_layer_mean_accu.append(accu)
        return per_layer_mean_accu

    def layer_mean_accu(self, layer_id):
        accu_dict = self.accu_history[layer_id]
        # return torch.from_numpy((np.sum([x for x in accu_dict.values()]) / len(accu_dict))
        return torch.mean(torch.concat([x.reshape(1, 1) for x in accu_dict.values()]))


class TokenPredState:
    def __init__(self):
        self.accu_history = {}
        self.catch_size = 1000  # estimate mean accuracy for the latest 1000 examples

    def update(self, layer_id, image_id, accu):
        if layer_id not in self.accu_history:
            self.accu_history[layer_id] = accu.reshape(-1, 1)
        elif len(self.accu_history[layer_id]) < self.catch_size:
            self.accu_history[layer_id] = torch.cat(
                (self.accu_history[layer_id], accu.reshape(-1, 1))
            )
        else:
            self.accu_history[layer_id] = torch.cat(
                (self.accu_history[layer_id][1:], accu.reshape(-1, 1))
            )
        # torch.cat((tensor[1:], Tensor([x])))
        # self.accu_history[layer_id][image_id.cpu().item()] = accu  # .cpu().item()

    def batch_update(self, layer_id, image_ids, accu_list):
        for img_id, acc in zip(image_ids, accu_list):
            self.update(layer_id=layer_id, image_id=img_id, accu=acc)

    @property
    def mean_accu(self):
        if len(self.accu_history) == 0:
            return 0.0

        per_layer_mean_accu = []
        for layer_id, _ in self.accu_history.items():
            accu = self.layer_mean_accu(layer_id)
            per_layer_mean_accu.append(accu)
        return per_layer_mean_accu

    def layer_mean_accu(self, layer_id):
        accu_tensor = self.accu_history[layer_id]
        # return torch.from_numpy((np.sum([x for x in accu_dict.values()]) / len(accu_dict))
        return torch.mean(accu_tensor)


class SetCriterion(nn.Module):

    def __init__(self, weight_dict, sgdt=None):
        super().__init__()

        self.weight_dict = weight_dict
        self.sgdt = sgdt
        self.token_scoring_loss = TokenPredLoss()
        self.token_train_accu_state = TokenPredState()
        self.token_test_accu_state = TokenPredState()

    def forward(self, outputs, sgdt, layer_ids=None):
        losses = {}

        sgdt_token_classification_output_list = outputs['sgdt_token_classification_output_list']
        if layer_ids is None:  # predict only the last layer
            layer_ids = list(range(len(sgdt_token_classification_output_list)))
            # layer_ids = [len(sgdt_token_classification_output_list) - 1]

        token_targets = sgdt.sgdt_targets['fg_gt']
        # sgdt_target_raw = self.sgdt.sgdt_target_raw
        valid_tokens = src_key_padding_mask2valid_token_mask(sgdt.src_key_padding_mask)

        for i, pred_logits in enumerate(sgdt_token_classification_output_list):
            if i not in layer_ids:
                continue

            l_dict = self.token_scoring_loss.cal_focal_loss_and_accu(
                pred_logits=pred_logits, token_targets=token_targets, valid_tokens=valid_tokens,
                cal_per_img_accu=True
            )

            # maintain the accu record
            accu_list = l_dict.pop('sgdt_fg_bg_accu')
            image_ids = [t['image_id'] for t in sgdt.targets]
            if self.training:
                self.token_train_accu_state.batch_update(layer_id=i, image_ids=image_ids, accu_list=accu_list)
                l_dict['sgdt_fg_bg_accu'] = self.token_train_accu_state.layer_mean_accu(i)
            else:
                self.token_test_accu_state.batch_update(layer_id=i, image_ids=image_ids, accu_list=accu_list)
                l_dict['sgdt_fg_bg_accu'] = self.token_test_accu_state.layer_mean_accu(i)

            l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
            losses.update(l_dict)

            if self.sgdt.token_adaption_visualization:
                outputs['sgdt_vis_data']['valid_tokens'] = valid_tokens
                outputs['sgdt_vis_data']['feat_map_size'] = sgdt.feat_map_size
                vis_tool = VisualizeToken(targets=sgdt.targets,
                                          sgdt_target_raw=sgdt.sgdt_target_raw,
                                          sgdt_targets=sgdt.sgdt_targets,
                                          sgdt_output=outputs['sgdt_vis_data']
                                          )
                vis_tool.visualize_token_adaption(sub_dir=self.sgdt.visualization_out_sub_dir)
            #     # vis_tool.save_intermediate_result()
            #     # vis_tool.save_significant_score_and_split_tokens(sub_dir=self.visualization_out_sub_dir)

        return losses

    # def forward(self, outputs, targets, src_key_padding_mask):
    #     self.sgdt.src_key_padding_mask = src_key_padding_mask  # B, H, W; to change to B, N use .flatten(1)
    #     bs, h, w = src_key_padding_mask.shape  # torch.Size([2, 25, 32])
    #     self.sgdt.feat_map_size = (h, w)
    #     self.sgdt.set_sgdt_targets(targets, feat_map_size=(h, w))
    #
    #     losses = {}  # sgdt loss
    #     sgdt_token_classification_output_list = outputs['sgdt_token_classification_output_list']
    #
    #     sgdt_targets = self.sgdt.sgdt_targets
    #     # sgdt_target_raw = self.sgdt.sgdt_target_raw
    #     valid_tokens = src_key_padding_mask2valid_token_mask(src_key_padding_mask)
    #
    #     for i, pred_logits in enumerate(sgdt_token_classification_output_list):
    #         l_dict = self.token_scoring_loss.cal_focal_loss_and_accu(
    #             pred_logits=pred_logits, token_targets=sgdt_targets, valid_tokens=valid_tokens
    #         )
    #         l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
    #         losses.update(l_dict)
    #
    #         # if self.sgdt.token_adaption_visualization:
    #         #     vis_tool = VisualizeToken(targets=targets,
    #         #                               sgdt_target_raw=sgdt_target_raw,
    #         #                               sgdt_targets=sgdt_targets,
    #         #                               sgdt_output=sgdt_output
    #         #                               )
    #         #     vis_tool.visualize_token_adaption(sub_dir=self.sgdt.visualization_out_sub_dir)
    #         #     # vis_tool.save_intermediate_result()
    #         #     # vis_tool.save_significant_score_and_split_tokens(sub_dir=self.visualization_out_sub_dir)
    #
    #     return losses


def build_token_classifier(args):
    device = torch.device(args.device)
    sgdt = build_sgdt(args)

    model = FgBgClassifier(embed_dim=args.hidden_dim, )

    weight_dict = {}
    # # ------------------------
    num_layer = get_num_of_layer(args.encoder_layer_config)
    if num_layer > 0:
        # Define, but not necessarily use.
        sgdt_weight_dict = dict(
            sgdt_loss_fg=1.0,  # args.sgdt_loss_fg_coef
            # sgdt_loss_token_significance=args.sgdt_loss_token_significance_coef,
            # sgdt_loss_small_scale=args.sgdt_loss_small_scale_coef,
        )
        for i in range(num_layer):
            weight_dict.update({k + f'_{i}': v for k, v in sgdt_weight_dict.items()})

    criterion = SetCriterion(sgdt=sgdt, weight_dict=weight_dict)
    criterion.to(device)

    return model, criterion, None
