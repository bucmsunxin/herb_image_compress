# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import pdb
import os.path as osp

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer, match_layer_names
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.config import integrate_cfg
from maskrcnn_benchmark.engine.transfer import do_transfer
from maskrcnn_benchmark.engine.cutter import Cutter, cut_load_model


def create_dir(tmp_dir):
    if not osp.exists(tmp_dir):
        os.makedirs(tmp_dir)


def set_transfer_output(cfg):
    output_dir = cfg.OUTPUT_DIR
    transfer_method = cfg.MODEL.TRANSFER.METHOD
    output_name = cfg.MODEL.TRANSFER.OUTPUT_NAMES[0]
    weight_value = cfg.MODEL.TRANSFER.WEIGHT_VALUES[0]
    output_dir = '%s/TRANSFER_%s_%.1f' % (output_dir, output_name, weight_value)
    return output_dir


def train(cfg_list, cfg_names, local_rank, distributed):
    # building and loading models
    model_list = []
    for cfg, cfg_name in zip(cfg_list, cfg_names):
        model = build_detection_model(cfg)

        # setting up loading check-pointer
        if is_main_process():
            print(model)
        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)
        model_list.append(model)
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.load()
    synchronize()

    # only use student config for training
    cfg = cfg_list[-1]
    cfg.SOLVER.BASE_LR = cfg.CUT_SOLVER.BASE_LR
    cfg.SOLVER.WARMUP_FACTOR = cfg.CUT_SOLVER.WARMUP_FACTOR
    cfg.SOLVER.WARMUP_ITERS = cfg.CUT_SOLVER.WARMUP_ITERS
    cfg.SOLVER.STEPS = cfg.CUT_SOLVER.STEPS
    cfg.SOLVER.MAX_ITER = cfg.CUT_SOLVER.MAX_ITER
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.CUT_SOLVER.CHECKPOINT_PERIOD

    # distribute student and teacher models
    model = model_list[-1]

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    synchronize()

    model_T = model_list[0]
    model_T.eval()
    if distributed:
        model_T = torch.nn.parallel.DistributedDataParallel(
            model_T, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    synchronize()

    # load TRANSFERed model for prunning initialization
    output_dir = cfg.OUTPUT_DIR
    #output_dir = set_transfer_output(cfg)
    assert(os.listdir(output_dir)), 'transferred model dir does not exist!'
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    assert('last_checkpoint' in os.listdir(output_dir))
    checkpointer.load()


#############################################################################################
    ## setup TRANSFER
    model_prefix = cfg.MODEL.TRANSFER.MODEL_PREFIX
    output_names = cfg.MODEL.TRANSFER.OUTPUT_NAMES
    output_names = [model_prefix + out_name for out_name in output_names]
    transfer_layers = {
         'large': output_names,
         'small': output_names,
    }
    weight_values = cfg.MODEL.TRANSFER.WEIGHT_VALUES
    transfer_weights = [[layer, weight] for layer, weight in zip(output_names, weight_values)]
    transfer_method = cfg.MODEL.TRANSFER.METHOD

    ## setup CUT
    curt_layers = cfg.MODEL.CUT.CURT_LAYERS
    next_layers = cfg.MODEL.CUT.NEXT_LAYERS
    cut_percents = cfg.MODEL.CUT.CUT_PERCENTS
    cut_names = cfg.MODEL.CUT.CUT_NAMES
    num_layers = len(curt_layers)

    output_dir = '%s/CUT_' % output_dir
    stage_percents =  cfg.MODEL.CUT.MOBILENETV2.CUSTOM_CUT_PERCENTS
    out_name = ''
    for stage_percent in stage_percents:
        out_name += '_%.1f' % stage_percent
    output_dir += '%s/' % out_name
    create_dir(output_dir)
    dataset = cfg.MODEL.DATASET

    for layer_idx in range(num_layers):
        curt_layer = curt_layers[-1-layer_idx]
        next_layer = next_layers[-1-layer_idx]
        cut_percent = cut_percents[-1-layer_idx]
        cut_name = cut_names[-1-layer_idx]

        if layer_idx == 0:
            output_dir += '%.1f' % cut_percent
        else:
            output_dir += '_%.1f' % cut_percent
        cfg.OUTPUT_DIR = output_dir
        if is_main_process():
            create_dir(output_dir)
        synchronize()

        # train or load?
        if 'last_checkpoint' in os.listdir(output_dir):
            print('############ Loading %d / %d : currect %s -> next %s , cut %s: %.1f ###################' % \
                              (num_layers-layer_idx, num_layers, curt_layer, next_layer, cut_name, cut_percent))
            if distributed:
                model = model.module
            model.eval()
            model = cut_load_model(model, curt_layer, next_layer, cut_percent)
            if distributed:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False,
                )
            checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
            checkpointer.load()
            synchronize()
            continue

        # setup data_loader for prunning
        print('############ Cutting %d / %d : current %s -> next %s , cut %s: %.1f ###################' % \
                      (num_layers-layer_idx, num_layers, curt_layer, next_layer, cut_name, cut_percent))
        cfg.MODEL.CUT.IS_OPEN = True
        cfg.DATASETS.TEST = ("%s_val"%dataset,)
        data_loader = make_data_loader(
            cfg,
            is_train=False,
            is_distributed=distributed,
            start_iter=0,
        )

        if distributed:
            model = model.module
        model.eval()
        with torch.no_grad():
            cutter = Cutter(model, data_loader, curt_layer, next_layer, cut_percent)
            model = cutter.cut()
            if is_main_process():
                print(model)
        synchronize()

        arguments = {}
        arguments["iteration"] = 0
        model.train()
        synchronize()
        optimizer = make_optimizer(cfg, model)
        scheduler = make_lr_scheduler(cfg, optimizer)
        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                  model, device_ids=[local_rank], output_device=local_rank,
                 # this should be removed if we update BatchNorm stats
                 broadcast_buffers=False,
            )
        synchronize()


        checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, output_dir, save_to_disk)
        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

        cfg.MODEL.CUT.IS_OPEN = False
        data_loader = make_data_loader(
             cfg,
             is_train=True,
             is_distributed=distributed,
             start_iter=0,
        )

        # training
        do_transfer(
             model_T,
             model,
             data_loader,
             optimizer,
             scheduler,
             checkpointer,
             device,
             checkpoint_period,
             arguments,
             transfer_layers,
             transfer_weights,
             transfer_method,
             cfg,
        )

        #### testing
        synchronize()
        model.eval()
        cfg.DATASETS.TEST = ("%s_val"%dataset,)
        test(cfg, model, distributed)
        synchronize()

    return model


def test(cfg, model, distributed):
    # if distributed:
    #     model = model.module
    model.eval()
    torch.cuda.empty_cache()  # TODO check if it helps
    output_folders = [None] * len(cfg.DATASETS.TEST)
    if cfg.OUTPUT_DIR:
        dataset_names = cfg.DATASETS.TEST
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    data_loaders = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, data_loader_val in zip(output_folders, data_loaders):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            device=cfg.MODEL.DEVICE,
            output_folder=output_folder,
            cfg=cfg,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-large-file",
        dest="config_large_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--config-small-file",
        dest="config_small_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    # specify configuration for each model
    cfg_t = cfg.clone()
    cfg_s = cfg.clone()
    cfg_t.merge_from_file(args.config_large_file)
    cfg_s.merge_from_file(args.config_small_file)
    cfg_t.merge_from_list(args.opts)
    cfg_s.merge_from_list(args.opts)

    cfg_s.MODEL.TRANSFER.IS_OPEN = True
    cfg_s.MODEL.CUT.IS_OPEN = True
    integrate_cfg(cfg_t)
    integrate_cfg(cfg_s)

    cfg_file = args.config_small_file
    output_dir = cfg_s.OUTPUT_DIR
    assert(os.listdir(output_dir)), 'TRANSFER model dir does not exist!'

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(cfg_file))
    with open(cfg_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg_s))

    cfg_list = [cfg_t, cfg_s]
    cfg_names = ['large', 'small']
    model = train(cfg_list, cfg_names, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()
