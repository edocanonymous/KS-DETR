import os


def get_out_dir_from_file_name(abs_config_file_name, with_parent_dir=False):
    file_name = os.path.basename(abs_config_file_name).split('.')[0]
    if with_parent_dir:
        parent_dir = os.path.basename(os.path.dirname(abs_config_file_name))
        return f"./output/{parent_dir}/{file_name}"
    else:
        return f"./output/{file_name}"

# https://detrex.readthedocs.io/en/latest/tutorials/Download_Pretrained_Weights.html
R101_WEIGHT_PATH = "output/weights/R-101.pkl"
R50_WEIGHT_PATH = "output/weights/R-50.pkl"
SWIN_TINY_WEIGHT_PATH = "output/weights/swin_tiny_patch4_window7_224_22kto1k_finetune.pth"
SWIN_SMALL_WEIGHT_PATH = "output/weights/swin_small_patch4_window7_224_22kto1k_finetune.pth"
SWIN_BASE_WEIGHT_PATH = "output/weights/swin_base_patch4_window7_224_22kto1k.pth"


# https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth

# Use the following two because detrex used these in their config, not used the above.
# https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22kto1k_finetune.pth
# https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22kto1k_finetune.pth
# https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth