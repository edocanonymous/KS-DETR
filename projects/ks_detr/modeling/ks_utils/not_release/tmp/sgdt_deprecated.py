from torch import nn

from projects.ks_detr.modeling import KSGT

SIGMA = 0.05


class GTRatioOrSigma:
    def __init__(self, gt_decay_criterion,  # default=None
                 data_size, total_epoch, decay_sigma=False
                 ):
        # for gt proposal fusion
        self.gt_decay_criterion = gt_decay_criterion
        self.data_size = data_size
        self.total_epoch = total_epoch
        self.decay_start_epoch = None
        self.decay_end_epoch = None
        self.total_steps = None
        self.gt_ratio_updater_ready = False
        self.gt_ratio = 1.0

        if self.gt_decay_criterion is not None:
            self.decay_start_epoch = 0
            self.decay_end_epoch = total_epoch

            if gt_decay_criterion != '':
                decay_start_epoch = extract_thd(gt_decay_criterion, 'start_epoch')
                if decay_start_epoch is not None:
                    assert isinstance(decay_start_epoch, (int, float)) and decay_start_epoch >= 0
                    self.decay_start_epoch = decay_start_epoch

                decay_end_epoch = extract_thd(gt_decay_criterion, 'end_epoch')
                if decay_end_epoch is not None:
                    assert isinstance(decay_end_epoch, (int, float)) and decay_end_epoch > 0
                    self.decay_end_epoch = decay_end_epoch
            assert self.decay_start_epoch < self.decay_end_epoch < self.total_epoch

            self.total_steps = self.data_size * (self.decay_end_epoch - self.decay_start_epoch)
            self.gt_ratio_updater_ready = True
        print(f'self.gt_ratio_updater_ready = {self.gt_ratio_updater_ready}')

        self.sigma_max = SIGMA
        self.sigma = SIGMA
        self.decay_sigma = decay_sigma

    def update_gt_ratio(self, cur_epoch, cur_iter):
        if self.gt_decay_criterion is not None:
            if cur_epoch < self.decay_start_epoch:
                self.gt_ratio = 1.0
            elif cur_epoch >= self.decay_end_epoch:
                self.gt_ratio = 0
            else:
                cur_step = (cur_epoch - self.decay_start_epoch) * self.data_size + cur_iter
                self.gt_ratio = 1 - cur_step / self.total_steps

    def update_sigma(self, cur_epoch, cur_iter):
        if self.decay_sigma:
            total_steps = self.data_size * self.total_epoch
            cur_step = cur_epoch * self.data_size + cur_iter
            process = cur_step / total_steps
            sigma_multiplier = 1 - process
            self.sigma = SIGMA * sigma_multiplier

    def update(self, cur_epoch, cur_iter):
        self.update_gt_ratio(cur_epoch=cur_epoch, cur_iter=cur_iter)
        self.update_sigma(cur_epoch=cur_epoch, cur_iter=cur_iter)


def init_proposal_processor(proposal_scoring):
    if proposal_scoring is not None:
        proposal_scoring_parser = ProposalScoringParser(proposal_scoring_config=proposal_scoring)
        proposal_filtering_param = proposal_scoring_parser.extract_box_filtering_parameter()
        proposal_processor = ProposalProcess(**proposal_filtering_param)
    else:
        proposal_processor = ProposalProcess()

    return proposal_processor


class TokenClassifier(nn.Module):

    def __init__(self, embed_dim, num_class=2):  # channel dim, also is the feature dimension for each token
        super().__init__()
        assert isinstance(embed_dim, int)

        hidden_dim = embed_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_class),  # 3 classes.
            # nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, feat_map_size=None, with_global_feat=False, mask=None):
        """ Significant value prediction, < 0.5, bg, > 0.5 fg (smaller object, large significance).
        Args:
            x: dim: N, B, C, where N is the number of tokens, B is the batch size,
            C is the channel dimension.
        Returns:
        """
        # torch.Size([750, 2, 256])
        return self.mlp(x)  # torch.Size([650, 2, 3])


class FgBgClassifier(nn.Module):
    """ This is the DAB-DETR module that performs object detection """

    def __init__(self, embed_dim, ):
        super().__init__()

        self.token_classifier = TokenClassifier(embed_dim=embed_dim)

    def forward(self, teacher_encoder_output_list, ):
        classification_logits = []
        for k, teacher_encoder_output in enumerate(teacher_encoder_output_list):
            x = teacher_encoder_output['feat']
            pred = self.token_classifier(x)
            classification_logits.append(pred)  # torch.Size([750, 2, 2])

        out = {'ksgt_token_classification_output_list': classification_logits}

        # ================ Visualization purpose
        softmax = nn.Softmax(dim=-1)
        prob = softmax(classification_logits[-1])[:, -1]
        tokens_to_split = classification_logits[-1].max(dim=-1).indices
        tokens_to_discard = 1 - tokens_to_split
        vis_data = {
            'tokens_small_obj': tokens_to_split,
            'tokens_to_discard': tokens_to_discard,
            # 'valid_tokens': valid_tokens_float,  # TODO: change this to bool()

            # valid_tokens in the original size, this is only used for loss calculation.
            # 'valid_tokens_original': valid_tokens_original_float,
            'tokens_to_discard_original': tokens_to_discard,
            'tokens_to_split_original': tokens_to_split,
            'fg_score': prob,
            'small_scale_score': prob,
        }
        out.update(dict(ksgt_vis_data=vis_data))
        return out


def build_ksgt(args):
    return KSGT(args)