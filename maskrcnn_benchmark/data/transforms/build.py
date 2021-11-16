# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T
import cv2
import numpy as np
import random
from PIL import Image


def build_transforms(cfg, is_train=True):
    input_size = cfg.INPUT.MAX_SIZE_TRAIN
    test_scale_size = 256

    if is_train:
        transform = T.Compose(
            [
                T.RandomSizedCrop(input_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Scale(test_scale_size),
                T.CenterCrop(input_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transform