# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n, means, stds):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", means)
        self.register_buffer("running_var", stds)

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


class LearnedBatchNorm2d(nn.BatchNorm2d):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    can be learned
    """
    def __init__(self, n):
        super(LearnedBatchNorm2d, self).__init__(n)

    def forward(self, x):
        return super(LearnedBatchNorm2d, self).forward(x)

