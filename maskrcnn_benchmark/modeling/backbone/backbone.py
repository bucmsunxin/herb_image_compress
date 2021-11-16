# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from . import resnet
from . import mobilenet_v2
import torchvision

def build_backbone(cfg, body):
    pretrained = cfg.MODEL.USE_PRETRAIN
    if body.startswith('X-101'):
        model = resnet.resnext101_32x8d(pretrained=pretrained)
    elif body.startswith('R-50'):
        model = resnet.resnet50(pretrained=pretrained)
    elif body.startswith('Mobile-V2'):
        model = mobilenet_v2.mobilenet_v2(pretrained=pretrained)
    else:
        raise RuntimeError('not supported body %s' % body)
    return model
