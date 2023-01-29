#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys
import time
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger("detrex")

# =====================
import logging
from collections.abc import Mapping
import smrc.utils
from collections import OrderedDict

def save_csv_format(results, save_prefix):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.

    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
            unordered dict can also be printed, but in arbitrary order
    """
    assert isinstance(results, Mapping) or not len(results), results
    # logger = logging.getLogger(__name__)


    """
    [01/23 17:52:57 d2.evaluation.testing]: copypaste: Task: bbox
    [01/23 17:52:57 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
    [01/23 17:52:57 d2.evaluation.testing]: copypaste: 44.7465,63.9818,48.8243,26.9734,47.9329,59.7535   
    """
    for task, res in results.items():
        result = []
        if isinstance(res, Mapping):
            # Don't print "AP-category" metrics since they are usually not tracked.
            important_res = [(k, v) for k, v in res.items() if "-" not in k]
            # logger.info("copypaste: Task: {}".format(task))
            result.append([",".join([k[0] for k in important_res])])
            # result.append([",".join(["{0:.4f}".format(k[1]) for k in important_res])])
            result.append([k[1] for k in important_res])
        else:
            # logger.info(f"copypaste: {task}={res}")
            result.append(res)

        smrc.utils.save_multi_dimension_list_to_file(
            filename=f'{save_prefix}.txt',
            list_to_save=result
        )

        # res = smrc.utils.load_multi_column_list_from_file(
        #     filename=f'{save_prefix}.txt',
        # )
        print(f'================== {save_prefix}.txt \n {result} ')
    # for task, res in results.items():
    #     if isinstance(res, Mapping):
    #         # Don't print "AP-category" metrics since they are usually not tracked.
    #         important_res = [(k, v) for k, v in res.items() if "-" not in k]
    #         logger.info("copypaste: Task: {}".format(task))
    #         logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
    #         logger.info("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))
    #     else:
    #         logger.info(f"copypaste: {task}={res}")
# =====================


class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
            self.grad_scaler = grad_scaler

        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        with autocast(enabled=self.amp):
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )

        print_csv_format(ret)

        return ret

def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
    )

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

        # model.ksgt.eval_decoder_layer = 5
        # model.ksgt.save_eval_result_file_prefix = 374999
        # save_prefix = None
        # if hasattr(model, 'ksgt') and model.ksgt.eval_and_save_teacher_result:
        checkpoint_file = cfg.train.init_checkpoint
        iter = str(int(os.path.basename(checkpoint_file).split('.')[0].replace('model_', '')) + 1)  # model_0374999.pth
        result_dir=os.path.dirname(cfg.train.init_checkpoint)
        save_prefix = os.path.join(
            result_dir,   # cfg.train.output_dir,
            f'ap_e{cfg.model.ksgt.eval_decoder_layer}_{iter}iter')
        # ret = OrderedDict([('bbox', {'AP': 50.393421095419036, 'AP50': 67.62396478644013, 'AP75': 55.17506819573489, 'APs': 33.185357202806884, 'APm': 55.51752934577829, 'APl': 62.26931490693006, 'AP-person': 61.94011561294711, 'AP-bicycle': 39.45589804781215, 'AP-car': 50.00944143150199, 'AP-motorcycle': 53.19816841611469, 'AP-airplane': 78.58020680878397, 'AP-bus': 74.49894712118606, 'AP-train': 80.65240581969287, 'AP-truck': 47.47029310368717, 'AP-boat': 49.421602863872025, 'AP-traffic light': 42.40273109335477, 'AP-fire hydrant': 74.42781423739383, 'AP-stop sign': 77.90895367629761, 'AP-parking meter': 52.53989465062916, 'AP-bench': 39.27746197295666, 'AP-bird': 51.674060645953425, 'AP-cat': 77.84712829439677, 'AP-dog': 69.06743870094473, 'AP-horse': 66.95950643910649, 'AP-sheep': 60.15803532267932, 'AP-cow': 66.97777771860885, 'AP-elephant': 75.073299048648, 'AP-bear': 85.8068349899908, 'AP-zebra': 77.56148945889409, 'AP-giraffe': 80.20254840970563, 'AP-backpack': 17.033801939591992, 'AP-umbrella': 52.7028684044937, 'AP-handbag': 15.904406251502925, 'AP-tie': 32.07915771899298, 'AP-suitcase': 44.642852447483975, 'AP-frisbee': 65.10301931342639, 'AP-skis': 37.68191942536976, 'AP-snowboard': 33.97199431997907, 'AP-sports ball': 43.99789530551806, 'AP-kite': 58.44308997652982, 'AP-baseball bat': 35.230716668358625, 'AP-baseball glove': 33.990712056334026, 'AP-skateboard': 58.579901765772405, 'AP-surfboard': 48.77670534138752, 'AP-tennis racket': 52.75078183398871, 'AP-bottle': 45.65569682711798, 'AP-wine glass': 36.43769605586516, 'AP-cup': 44.29234992022484, 'AP-fork': 29.346588957025375, 'AP-knife': 17.00817153349032, 'AP-spoon': 20.483287623291144, 'AP-bowl': 44.200548844077616, 'AP-banana': 40.32402972674715, 'AP-apple': 28.96271436320788, 'AP-sandwich': 38.34322609851323, 'AP-orange': 36.07441762207906, 'AP-broccoli': 35.95121550406633, 'AP-carrot': 24.082137810829256, 'AP-hot dog': 30.636954261661685, 'AP-pizza': 52.109712822347866, 'AP-donut': 59.22009764611856, 'AP-cake': 35.300495932900176, 'AP-chair': 37.18614131564514, 'AP-couch': 55.12624517779061, 'AP-potted plant': 54.415049138317606, 'AP-bed': 66.1725245303208, 'AP-dining table': 51.13771963029365, 'AP-toilet': 78.09591262354144, 'AP-tv': 72.7086362212489, 'AP-laptop': 60.80641814344235, 'AP-mouse': 65.3674530487707, 'AP-remote': 29.997744112779472, 'AP-keyboard': 55.45684900646592, 'AP-cell phone': 32.50596941471353, 'AP-microwave': 63.08616480672623, 'AP-oven': 56.2809076644498, 'AP-toaster': 55.56733520923282, 'AP-sink': 63.60052851483594, 'AP-refrigerator': 71.75828312206792, 'AP-book': 25.184763228133384, 'AP-clock': 67.59718886212197, 'AP-vase': 53.25065058310958, 'AP-scissors': 28.732171941823687, 'AP-teddy bear': 55.44818852019438, 'AP-hair drier': 26.696409197896497, 'AP-toothbrush': 22.86321541614919})])
        ret = do_test(cfg, model)
        if save_prefix:
            save_csv_format(ret, save_prefix)

        # print(ret)

    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
