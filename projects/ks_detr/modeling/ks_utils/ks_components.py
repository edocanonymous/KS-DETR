import torch


def parser_encoder_decoder_layers(encoder_decoder_layer_config):
    layer_conf_split = encoder_decoder_layer_config.split('-')
    encoder_layer_conf_list = []
    for l_conf in layer_conf_split:
        l_type_and_num = l_conf.split('_')
        assert len(l_type_and_num) == 2, f'The format of encoder layer config is wrong, ' \
                                         'expected length 2, e.g., regular_6, but got' \
                                         '{l_conf}'
        l_type, num_l = l_type_and_num[0], int(l_type_and_num[1])
        # assert l_type in ['regular', 'ksgt', 'ksgtv1'] and num_l > 0
        assert num_l > 0
        encoder_layer_conf_list.append([l_type, num_l])
    return encoder_layer_conf_list


# def extract_encoder_layer_config():


def is_with_ksgt_layer(encoder_layer_config):
    if encoder_layer_config is None:
        return False
    return encoder_layer_config.find('ksgt') > -1


def get_num_ksgt_layer(encoder_layer_config):
    encoder_layer_list = parser_encoder_decoder_layers(encoder_layer_config)
    cnt = 0
    for (l_type, num_l) in encoder_layer_list:
        # if l_type in ['ksgt', 'ksgtv1']:
        if is_str_exist(l_type, 'ksgt'):
            cnt += num_l
    return cnt


def get_num_of_layer(encoder_layer_config):
    encoder_layer_list = parser_encoder_decoder_layers(encoder_layer_config)
    cnt = 0
    for (l_type, num_l) in encoder_layer_list:
        cnt += num_l
    return cnt


def split_encoder_layer(encoder_layer_config):
    encoder_layer_list = parser_encoder_decoder_layers(encoder_layer_config)
    normal_layer_ids = []
    ksgt_layer_ids = []

    cnt = 0
    for (l_type, num_l) in encoder_layer_list:
        if is_str_exist(l_type, 'ksgt'):
            ksgt_layer_ids.extend(list(range(cnt, cnt + num_l)))
        else:
            normal_layer_ids.extend(list(range(cnt, cnt + num_l)))
            cnt += num_l
    return normal_layer_ids, ksgt_layer_ids


def is_str_exist(full_str, sub_str):
    return full_str.find(sub_str) > -1


def extract_thd(config_str, sub_str):
    # debug_gt_split_ratio
    assert len(sub_str) > 0

    configs = config_str.split('-')
    thd = None
    for v in configs:
        if v.find(sub_str) > -1:
            thd = float(v.split(sub_str)[-1])
            break

    return thd


class SGDTConfigParse:
    def __init__(self, config_str):
        self.config_str = config_str

    def str_exist(self, sub_str):
        return is_str_exist(self.config_str, sub_str)

    def _str_list_exist(self, sub_str_list):
        assert isinstance(sub_str_list, list)
        return [self.str_exist(sub_str) for sub_str in sub_str_list]

    def extract_thd(self, sub_str):
        return extract_thd(self.config_str, sub_str)

    def extract_sub_setting(self, sub_str):
        # debug_gt_split_ratio
        assert len(sub_str) > 0

        configs = self.config_str.split('-')
        setting = None
        for v in configs:
            if v.find(sub_str) > -1:
                assert v.split(':')[0] == sub_str
                setting = v.split(':')[-1]
                break

        return setting


def pad2d(instance_mask, padded_img_size):
    # # the regions to update is not a rectangle any more, but a binary mask.
    # instance_mask_h, instance_mask_w = instance_mask.shape
    # instance_mask_padded = torch.full(padded_img_size, False, device=instance_mask.device)
    # instance_mask_padded[:instance_mask_h, :instance_mask_w] = instance_mask

    padding_bottom, padding_right = padded_img_size[0] - instance_mask.shape[0], \
                                    padded_img_size[1] - instance_mask.shape[1]
    m = torch.nn.ZeroPad2d((0, padding_right, 0, padding_bottom))
    instance_mask_padded1 = m(instance_mask.float().unsqueeze(0)).bool().squeeze(0)
    # assert torch.equal(instance_mask_padded, instance_mask_padded1)
    return instance_mask_padded1


def src_key_padding_mask2valid_token_mask(src_key_padding_mask):
    assert src_key_padding_mask.dim() == 3  # B, H, W
    valid_tokens = ~(src_key_padding_mask.flatten(1).permute(1, 0))  # (B, N) -> (N, B)
    return valid_tokens
