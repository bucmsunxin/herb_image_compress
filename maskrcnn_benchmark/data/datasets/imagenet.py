# -*- coding: utf-8 -*-
import cv2
import pickle
import torch
import numpy as np
import math
import json
import random
import os
import os.path as osp
import pdb
import copy
from PIL import Image

def create_dir(tmp_dir):
    if not osp.exists(tmp_dir):
        os.makedirs(tmp_dir)

def load_file(data_file):
    ptr = open(data_file, 'rb')
    data = pickle.load(ptr)
    return data

def save_file(data, data_file):
    ptr = open(data_file, 'wb')
    pickle.dump(data, ptr)
    ptr.close()


class ImageNetDataset(object):
    def __init__(self, root, ann_file, transforms=None):
        self.root = root
        self.dataset = load_file(ann_file)
        self.transforms = transforms

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        imfile = '%s/%s' % (self.root, entry['imfile'])
        assert(osp.exists(imfile))
        img = Image.open(imfile).convert('RGB')

        # to tensor
        label = entry['label']
        target = torch.tensor(label)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target, idx

    def __len__(self):
        return len(self.dataset)

