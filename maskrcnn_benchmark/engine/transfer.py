# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import pdb
import os
import os.path as osp

import torch
import torch.distributed as dist
import torch.nn.functional as F

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.structures.image_list import to_image_list

def create_dir(tmp_dir):
    if not osp.exists(tmp_dir):
        os.makedirs(tmp_dir)


def register_hook(model, layer_names, feats_dict, cfg):
    """
    register hook on given transfer_layers of input model to save their output feature
    maps to transfer_feats_dict.
    :param model: nn.Module, model to register hook on
    :param layer_names: list[layer_name], where to register hooks
    :param feats_dict: dict[layer_name -> feature_map], where to save feature maps
    :return: list[hook_handle], all hook handles registered
    """

    def get_hook(feats_dict, key):
        def hook(self, input, output):
            feats_dict[key] = output
        return hook

    hook_handles = []
    modules = dict(model.named_modules())
    for layer_name in layer_names:
        #assert layer_name in modules, f"can not found {layer_name} in input model, please check!"
        layer = modules[layer_name]
        hook_handles.append(layer.register_forward_hook(get_hook(feats_dict, layer_name)))
    return hook_handles


def append_transfer_loss(model_T, model_S, transfer_feats_dict, transfer_weights_dict, loss_dict, transfer_method, cfg):
    for name, weight in transfer_weights_dict.items():

        feat_T = F.sigmoid(transfer_feats_dict['large'][name])
        feat_S = F.sigmoid(transfer_feats_dict['small'][name])
        transfer_loss = F.mse_loss(feat_S, feat_T, size_average=False)

        num_img = feat_T.shape[0]
        transfer_loss *= weight / num_img
        loss_dict['loss_transfer'] = transfer_loss
    return



def transfer_targets(model_T, model_S, images, targets, transfer_feats_dict, transfer_weights, transfer_method, arch, cfg):
    # teacher
    with torch.no_grad():
        output = model_T(images)
        
    loss_dict = model_S(images, targets)

    transfer_weights_dict = {}
    for transfer_weight in transfer_weights:
        transfer_weights_dict[transfer_weight[0]] = transfer_weight[1]
    append_transfer_loss(model_T, model_S, transfer_feats_dict, transfer_weights_dict, loss_dict, transfer_method, cfg)

    return loss_dict


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            all_losses.append(v)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_transfer(
    model_T,
    model_S,
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
):
    #pdb.set_trace()
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]

    model_T.eval()
    model_S.train()

    transfer_feats_dict = {'large': {}, 'small': {}}
    register_hook(model_T, transfer_layers['large'], transfer_feats_dict['large'], cfg)
    register_hook(model_S, transfer_layers['small'], transfer_feats_dict['small'], cfg)

    start_training_time = time.time()
    end = time.time()
    arch = cfg.MODEL.META_ARCHITECTURE
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        targets = torch.tensor(targets).to(device)

        loss_dict = transfer_targets(model_T, model_S, images, targets, transfer_feats_dict, transfer_weights, transfer_method, arch, cfg)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == (max_iter - 1):
        #if iteration % checkpoint_period == 0 or iteration == (max_iter - 1):
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0 and iteration > 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
