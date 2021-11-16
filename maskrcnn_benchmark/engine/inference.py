# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import tempfile
import time
import os
import os.path as osp
from collections import OrderedDict
import sys

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
import pdb
import cv2
import numpy as np

from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str


def compute_on_dataset(model, data_loader, device):
    model.eval()
    preds_dict = {}
    gt_dict = {}
    cpu_device = torch.device("cpu")
    for k, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            output = model(images.to(device))
            output = [o.to(cpu_device) for o in output]
        preds_dict.update(
            {img_id: result.topk(5)[1] for img_id, result in zip(image_ids, output)}
        )
        gt_dict.update(
            {img_id: result for img_id, result in zip(image_ids, targets)})
    return preds_dict, gt_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        count_iter=None,
        device="cuda",
        output_folder=None,
        cfg=None,
):
    #convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    total_timer.tic()
    preds, gts = compute_on_dataset(model, data_loader, device)

    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    preds = _accumulate_predictions_from_multiple_gpus(preds)
    synchronize()
    gts = _accumulate_predictions_from_multiple_gpus(gts)
    synchronize()
   
    if not is_main_process():
        return

    if output_folder:
        preds_file = os.path.join(output_folder, "preds.pth")
        torch.save(preds, preds_file)

    if output_folder:
        gts_file = os.path.join(output_folder, "gts.pth")
        torch.save(gts, gts_file)

    ## define top1&top5 error
    if count_iter is None:
        result_file = os.path.join(output_folder, "result.log")
    else:
        result_file = os.path.join(output_folder, "result_%d.log" % count_iter)
    ptr = open(result_file, 'w')
    assert(len(preds) == len(gts))
    num_test = len(preds)
    acc1_cnt = 0
    acc5_cnt = 0

    ## define classification precision
    precision_file = os.path.join(output_folder, "class_precision.log")
    class_ptr = open(precision_file, 'w')
    num_classes = cfg.MODEL.NUM_CLASSES
    preds_classes = np.zeros((num_classes), dtype=np.int)
    gts_classes = np.zeros((num_classes), dtype=np.int)

    ## define confustion matrix
    cfm_file = os.path.join(output_folder, "confusion_matrix.log")
    cfm_ptr = open(cfm_file, 'w')
    preds_cls_for_cfm = []
    gts_cls_for_cfm = []

    for i, (pred, gt) in enumerate(zip(preds, gts)):
        assert(gt.item() < num_classes and pred[0].item() < num_classes)
        gts_classes[gt] += 1

        if gt in pred:
            acc5_cnt += 1
        if gt == pred[0]:
            acc1_cnt += 1
            preds_classes[gt] += 1

        # the class index in confusion matrix starts from 1
        preds_cls_for_cfm.append(pred[0].item()+1)
        gts_cls_for_cfm.append(gt+1)

    acc_top1 = float(acc1_cnt)/num_test * 100
    acc_top5 = float(acc5_cnt)/num_test * 100
    err_top1 = (1 - float(acc1_cnt)/num_test) * 100
    err_top5 = (1 - float(acc5_cnt)/num_test) * 100
    print('top1_err: %.3f' % err_top1)
    print('top5_err: %.3f' % err_top5)
    print('top1_acc: %.3f' % acc_top1)
    print('top5_acc: %.3f' % acc_top5)
    ptr.write('top1_acc: %.3f\n' % acc_top1)
    ptr.write('top5_acc: %.3f\n' % acc_top5)
    ptr.write('top1_err: %.3f\n' % err_top1)
    ptr.write('top5_err: %.3f\n' % err_top5)
    ptr.close()

    ## every class precision
    acc_classes = preds_classes.astype(np.float32) / gts_classes.astype(np.float32)
    for idx in range(num_classes):
        class_ptr.write('%d/%d\n' % (preds_classes[idx],gts_classes[idx]))
    class_ptr.write('\n')
    for idx in range(num_classes):
        class_ptr.write('%.4f\n' % acc_classes[idx])
    class_ptr.close()

    ## consufion matrix for pred_cls & gt_cls: starts from index 1
    cfm_ptr.write('prds_cls\n')
    for idx in range(num_test):
        cfm_ptr.write('%d,' % preds_cls_for_cfm[idx])
    cfm_ptr.write('\n')
    cfm_ptr.write('gt_cls\n')
    for idx in range(num_test):
        cfm_ptr.write('%d,' % gts_cls_for_cfm[idx])
    cfm_ptr.close()

    
