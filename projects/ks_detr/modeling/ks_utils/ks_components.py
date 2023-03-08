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


def get_num_of_layer(encoder_layer_config):
    encoder_layer_list = parser_encoder_decoder_layers(encoder_layer_config)
    cnt = 0
    for (l_type, num_l) in encoder_layer_list:
        cnt += num_l
    return cnt


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


