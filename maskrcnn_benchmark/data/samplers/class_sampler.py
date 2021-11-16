import math
import torch
from torch.utils.data.sampler import Sampler
import numpy as np
import random
import time

class ClassSampler(Sampler):
    def __init__(self, dataset, cfg):
        self.dataset = dataset
        self.epoch = 0
        self.num_images = len(self.dataset)
        self.cls_index_list = dataset.cls_index_list
        self.num_classes = len(dataset.cls_index_list)
        self.batch_size = 32
        self.cfg = cfg

    def __iter__(self):
        timestamp = time.localtime(time.time())
        tm_sec = timestamp.tm_sec * random.randint(1,100)
        g = torch.Generator()
        g.manual_seed(tm_sec)
        np.random.seed(tm_sec)
        random.seed(tm_sec)

        indices = []
        cls_randoms = torch.randperm(self.num_classes, generator=g).tolist()
        cls_indexes = cls_randoms[:self.batch_size]
        for cls_id in cls_indexes:
            image_list = self.cls_index_list[cls_id]
            image_idx  = random.choice(image_list)
            indices.append(image_idx)
        return iter(indices)


    def __len__(self):
        return self.num_classes


    def set_epoch(self, epoch):
        self.epoch = epoch
