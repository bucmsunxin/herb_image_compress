# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Code is copy-pasted exactly as in torch.utils.data.distributed,
# with a modification in the import to use the deprecated backend
# FIXME remove this once c10d fixes the bug it has
import math
import torch
#import torch.distributed.deprecated as dist
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
import numpy as np
import random

class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, cfg, num_replicas=None, rank=None, shuffle=True, is_train=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.num_images = len(self.dataset)
        self.num_samples = int(math.ceil(self.num_images * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.is_train = is_train
        self.shuffle = self.is_train
        self.cfg = cfg


    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(self.num_images, generator=g).tolist()
            np.random.seed(self.epoch)
            random.seed(self.epoch)
        else:
            indices = torch.arange(self.num_images).tolist()

        # add extra samples to make it evenly divisible
        if self.is_train:
            indices += indices[: (self.total_size - len(indices))]
            assert len(indices) == self.total_size
            # subsample
            offset = self.num_samples * self.rank
            indices = indices[offset : offset + self.num_samples]
            assert len(indices) == self.num_samples
        else:
            if self.cfg.MODEL.CUT.IS_OPEN:
                indices += indices[: (self.total_size - len(indices))]
                assert len(indices) == self.total_size
                # subsample
                offset = self.num_samples * self.rank
                indices = indices[offset : offset + self.num_samples]
                assert len(indices) == self.num_samples
            else:
                if self.rank < self.num_replicas - 1:
                    # subsample
                    offset = self.num_samples * self.rank
                    indices = indices[offset : offset + self.num_samples]
                    assert len(indices) == self.num_samples
                elif self.rank == self.num_replicas - 1:
                    # subsample
                    offset = self.num_samples * self.rank
                    indices = indices[offset : self.num_images]

        return iter(indices)


    def __len__(self):
        return self.num_samples


    def set_epoch(self, epoch):
        self.epoch = epoch
