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
from maskrcnn_benchmark.engine.transfer import do_transfer
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer, match_layer_names
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.config import integrate_cfg


def create_dir(tmp_dir):
    if not osp.exists(tmp_dir):
        os.makedirs(tmp_dir)

def set_transfer_output(cfg):
    output_dir = cfg.OUTPUT_DIR
    transfer_method = cfg.MODEL.TRANSFER.METHOD
    output_name = cfg.MODEL.TRANSFER.OUTPUT_NAMES[0]
    weight_value = cfg.MODEL.TRANSFER.WEIGHT_VALUES[0]
    output_dir += '%s/TRANSFER_%s_%.1f' % (output_dir, output_name, weight_value)
    return output_dir


def train(cfg_list, cfg_names, local_rank, distributed):
    # building and loading models
    model_list = []
    for cfg, cfg_name in zip(cfg_list, cfg_names):
        model = build_detection_model(cfg)
        if is_main_process():
            print(model)
        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)
        model_list.append(model)
    synchronize()

    # only use student config for training
    cfg = cfg_list[-1]
    cfg.SOLVER.BASE_LR = cfg.TRANSFER_SOLVER.BASE_LR
    cfg.SOLVER.WARMUP_FACTOR = cfg.TRANSFER_SOLVER.WARMUP_FACTOR
    cfg.SOLVER.WARMUP_ITERS = cfg.TRANSFER_SOLVER.WARMUP_ITERS
    cfg.SOLVER.STEPS = cfg.TRANSFER_SOLVER.STEPS
    cfg.SOLVER.MAX_ITER = cfg.TRANSFER_SOLVER.MAX_ITER
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TRANSFER_SOLVER.CHECKPOINT_PERIOD

    # distribute student and teacher models
    model = model_list[-1]
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    synchronize()

    model_T = model_list[0]
    if distributed:
        model_T = torch.nn.parallel.DistributedDataParallel(
            model_T, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    synchronize()

    # setting up checkpointer
    output_dir = set_transfer_output(cfg)
    create_dir(output_dir)
    cfg.OUTPUT_DIR = output_dir

    arguments = {}
    arguments["iteration"] = 0
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, output_dir, save_to_disk)

    extra_checkpoint_data = {}
    if 'last_checkpoint' in os.listdir(output_dir):
        extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    # setting up data_loader
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    ## training
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

    return model


def test(cfg, model, distributed):
    # if distributed:
    #     model = model.module
    synchronize()
    model = model.eval()
    torch.cuda.empty_cache()  # TODO check if it helps
    output_folders = [None] * len(cfg.DATASETS.TEST)
    if cfg.OUTPUT_DIR:
        dataset_names = cfg.DATASETS.TEST
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
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
    integrate_cfg(cfg_t)
    integrate_cfg(cfg_s)
    #cfg.freeze()

    cfg_file = args.config_small_file
    output_dir = cfg_s.OUTPUT_DIR
    create_dir(output_dir)
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

    test(cfg_s, model, args.distributed)


if __name__ == "__main__":
    main()
