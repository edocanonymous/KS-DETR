
# =====================
import logging
from collections.abc import Mapping
import smrc.utils
from collections import OrderedDict
import os
from projects.ks_detr.lib_conf import LIB_ROOT_DIR
def collect_result(result_dir, save_prefix):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    file_list = smrc.utils.get_file_list_in_directory(
        directory_path=result_dir, ext_str='.txt'
    )
    result = []
    for file_path in file_list:
        res = smrc.utils.load_multi_column_list_from_file(
            filename=file_path,
        )
        # '24ep/ap_e-1_155000iter.txt'
        iter = os.path.basename(file_path).split('.')[0]
        result.append([iter] + res[1])
        # if isinstance(res, Mapping):
        #     # Don't print "AP-category" metrics since they are usually not tracked.
        #     important_res = [(k, v) for k, v in res.items() if "-" not in k]
        #     # logger.info("copypaste: Task: {}".format(task))
        #     result.append([",".join([k[0] for k in important_res])])
        #     # result.append([",".join(["{0:.4f}".format(k[1]) for k in important_res])])
        #     result.append([k[1] for k in important_res])
        #
    smrc.utils.save_multi_dimension_list_to_file(
        filename=f'{save_prefix}.txt',
        list_to_save=result
    )

    # res = smrc.utils.load_multi_column_list_from_file(
    #     filename=f'{save_prefix}.txt',
    # )
    print(f'================== {save_prefix}.txt \n {result} ')


# 'detrex_results/ks_deformable_detr/triple_accu/-1'
dir_root_path = os.path.join(LIB_ROOT_DIR, 'detrex_results/ks_deformable_detr/triple_accu/6')

dir_list = smrc.utils.get_dir_list_recursively(dir_root_path)
for dir_path in dir_list:
    collect_result(result_dir=os.path.join(dir_root_path, dir_path),
                   save_prefix=os.path.join(dir_root_path, dir_path))