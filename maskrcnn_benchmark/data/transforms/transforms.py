# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import cv2
import torch
import torchvision
from torchvision.transforms import functional as F

import numpy as np
import types
from numpy import random
import os
import pdb
import math
from PIL import Image
import numbers


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        #assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        return F.resize(image, self.size, self.interpolation)


class RandomSizedCrop(object):
    def __init__(self, size):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        self.scale = (0.08, 1.0)
        self.ratio = (3. / 4., 4. / 3.)
        self.interpolation = Image.BILINEAR

    def __call__(self, image):
        width, height = _get_image_size(image)
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            #if 0 < w <= width and 0 < h <= height:
            if 0 < w < width and 0 < h < height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return F.resized_crop(image, i, j, h, w, self.size, self.interpolation)

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(self.ratio)):
            w = width
            h = int(round(w / min(self.ratio)))
        elif (in_ratio > max(self.ratio)):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2

        return F.resized_crop(image, i, j, h, w, self.size, self.interpolation)


class Scale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        return F.resize(image, self.size, self.interpolation)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, image):
        return F.center_crop(image, self.size)


class ToTensor(object):
    def __call__(self, image):
        return F.to_tensor(image)


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, image):
        return F.normalize(image, self.mean, self.std, self.inplace)

