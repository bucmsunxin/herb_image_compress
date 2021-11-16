# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import pdb
from torch.nn import functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone



class ClsNet_Mobile(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(ClsNet_Mobile, self).__init__()

        self.conv_body = cfg.MODEL.BACKBONE.CONV_BODY

        self.backbone = build_backbone(cfg, self.conv_body)

        self.cfg = cfg


    def forward(self, tensors):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[tensor]): ground-truth class label
        Returns:
            result (Tensor or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns Tensor contains scores after softmax.

        """

        class_logits = self.backbone(tensors)
        
        return F.softmax(class_logits, dim=1)
