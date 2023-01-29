

# # 12 epoch,
# max_iter = 90000
# milestone = [60000, 80000, 90000]
# lr = [1.0, 0.1, 0.01]


# # 50 epoch
# max_iter = 375000
# milestone = [300000,  375000]
# lr = [1.0, 0.1]


# # 70 epoch, decay at 60, 67
# max_iter = 375000
# milestones: [450000, 502500, 525000]
# values: [1.0, 0.1, 0.01]


# from 300000 for 50 epoch
# from 60000 from 12 epoch


# from detrex.config import get_config
# lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
# train.max_iter = 375000


# from detrex.config import get_config
# lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
# train.max_iter = 375000
#
#
#
# from detrex.config import get_config
# lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
# train.max_iter = 90000

# [150000, 180000]
# from detrex.config import get_config
# lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_24ep
# train.max_iter = 180000


# # [225000, 270000]
# from detrex.config import get_config
# lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_36ep
# train.max_iter = 270000
