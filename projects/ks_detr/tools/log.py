"""
Plotting utilities to visualize training logs.
"""

import argparse
import os
# import json
import smrc.utils
from projects.ks_detr.lib_conf import LIB_ROOT_DIR
# import pickle
import ast
# import re


def list_accu(dir_name):

    dir_name = os.path.join(LIB_ROOT_DIR, dir_name)
    if os.path.isdir(dir_name):
        dir_list = [dir_name] + smrc.utils.get_dir_list_recursively(dir_name,)
        for dir_path in dir_list:
            print(f'{dir_path} =========================-')
            metric_file = os.path.join(dir_path, 'metrics.json')
            result_file = os.path.join(dir_path, 'result.txt')
            if os.path.isfile(metric_file):
                print(f'|| metric_file = {metric_file} |||| ')
                out_list = extract_result(metric_file)
                smrc.utils.save_multi_dimension_list_to_file(
                    filename=result_file,
                    list_to_save=out_list,
                    delimiter=','
                )
                print(f'|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
                print(result_file)
                cmd = f'cat {result_file}'
                os.system(cmd)


def extract_result(metric_file):  #
    # metric_file = os.path.join(LIB_ROOT_DIR,
    #                            'output/ks_dn_detr_r50_smlp_qk_share_v_outproj_ffn/metrics.json')  #

    """
    # "bbox/AP": 40.01603343461165,
    "bbox/AP50": 61.26111903120145, "bbox/AP75": 42.954506191614186, "bbox/APl": 57.56400476387522,
     "bbox/APm": 43.85543701071388, "bbox/APs": 20.860293672953418, "iteration": 300000

    Args:
        metric_file:

    Returns:

    """
    result_data = []
    with open(metric_file, "r") as ins:
        for line in ins:
            data = ast.literal_eval(line)
            # print(data)  # ["Title"]
            if 'bbox/AP' in data:
                map = ("%.2f" % data['bbox/AP'])
                
                lr = data['lr'] if 'lr' in data else 999
                iteration = data['iteration'] if 'iteration' in data else -1

                # map = ("%.2f" % data['bbox/AP'])

                result_data.append([iteration, map, lr,
                                    ("%.2f" % data['bbox/AP50']),
                                    ("%.2f" % data['bbox/AP75']),
                                    ("%.2f" % data['bbox/APs']),
                                    ("%.2f" % data['bbox/APm']),
                                    ("%.2f" % data['bbox/APl'])
                                    ])
    return result_data


# list_accu('output/ks_dn_detr_r50_smlp_qk_share_v_outproj_ffn')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image to video')
    parser.add_argument('-i', '--result_dir', default='', type=str, help='Path to image directory')
    args = parser.parse_args()

    list_accu(dir_name=args.result_dir)

#     # python smrc/tools/frame2video_tool.py \
#     #   -i /home/sirius/datase
