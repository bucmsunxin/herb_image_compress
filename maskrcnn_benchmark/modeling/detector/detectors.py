# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .cls_net import ClsNet

_DETECTION_META_ARCHITECTURES = {"Cls": ClsNet}

def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
