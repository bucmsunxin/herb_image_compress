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


class MedDataset(object):
    def __init__(self, root, ann_file, transforms=None):
        self.root = root
        self.num_classes = 95
        self.roidb = load_file(ann_file)
        self.cls_index_list = self.get_cls_index_list()
        self.transforms = transforms

    def get_cls_index_list(self):
        cls_index_list_file = './cls_index_list.pkl'
        if not osp.exists(cls_index_list_file):
            cls_index_list = [[] for _ in range(self.num_classes)]
            for i, entry in enumerate(self.roidb):
                label = entry['label']
                assert(label < self.num_classes)
                assert(label >= 0)
                cls_index_list[label].append(i)
            save_file(cls_index_list, cls_index_list_file)
        else:
            cls_index_list = load_file(cls_index_list_file)
        return cls_index_list

    def __getitem__(self, idx):
        entry = self.roidb[idx]
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
        return len(self.roidb)

