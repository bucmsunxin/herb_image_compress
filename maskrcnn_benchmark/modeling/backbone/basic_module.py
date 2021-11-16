import torch
import torch.nn.functional as F
from torch import nn
from collections import namedtuple
from maskrcnn_benchmark.layers import FrozenBatchNorm2d, LearnedBatchNorm2d
BatchNorm2d = LearnedBatchNorm2d
import math
import pdb



def get_percent(tar_idx, begin_idx, ptr_idx, percent_list):
    if ptr_idx >= tar_idx and ptr_idx <= begin_idx:
        percent = percent_list[ptr_idx]
    else:
        percent = 1.0
    return percent


############################# mobilenet v2 core module WITH BN #########################
class FirstStem(nn.Module):
    def __init__(self, inp, oup, stride):
        super(FirstStem, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = BatchNorm2d(oup)
        self.relu1 = nn.ReLU6(False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x


class LastStem(nn.Module):
    def __init__(self, inp, oup):
        super(LastStem, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn1 = BatchNorm2d(oup)
        self.relu1 = nn.ReLU6(False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x


############################# mobilenet v2 core module WITHOUT BN #########################
class FirstStem_NOBN(nn.Module):
    def __init__(self, inp, oup, stride):
        super(FirstStem_NOBN, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=True)
        self.relu1 = nn.ReLU6(False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        return x


class LastStem_NOBN(nn.Module):
    def __init__(self, inp, oup):
        super(LastStem_NOBN, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, 1, 1, 0, bias=True)
        self.relu1 = nn.ReLU6(False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        return x


############################# 1x1 block ##############################
class Conv1x1Block(nn.Module):
    def __init__(self, in_channels, out_channels, last_relu=True):
        super(Conv1x1Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.last_relu = last_relu
        if self.last_relu:
            self.relu = nn.ReLU(False)

        for m in [self.conv]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.last_relu:
            x = self.relu(x)
        return x


class ConvBn1x1Block(nn.Module):
    def __init__(self, in_channels, out_channels, last_relu=True):
        super(ConvBn1x1Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = LearnedBatchNorm2d(out_channels)
        self.last_relu = last_relu
        if self.last_relu:
            self.relu = nn.ReLU(False)

        for m in [self.conv, self.bn]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
            else:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
                nn.init.constant_(m.running_var, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.last_relu:
            x = self.relu(x)
        return x



############################# 1x1 FC block ##############################
class FCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, last_relu=True):
        super(FCBlock, self).__init__()
        self.fc_embed = nn.Linear(in_channels, out_channels)
        self.last_relu = last_relu
        if self.last_relu:
            self.relu = nn.ReLU(False)

        for m in [self.fc_embed]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc_embed(x)
        if self.last_relu:
            x = self.relu(x)
        return x


class FCBnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, last_relu=True):
        super(FCBnBlock, self).__init__()
        self.fc_embed = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.last_relu = last_relu
        if self.last_relu:
            self.relu = nn.ReLU(False)

        for m in [self.fc_embed, self.bn]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
            else:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
                nn.init.constant_(m.running_var, 1)

    def forward(self, x):
        x = self.fc_embed(x)
        x = self.bn(x)
        if self.last_relu:
            x = self.relu(x)
        return x


################################## 3x3 block ################################
class Conv3x3Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, padding=1, last_relu=True):
        super(Conv3x3Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                    dilation=dilation, padding=dilation if dilation > 1 else padding)
        self.last_relu = last_relu
        if self.last_relu:
            self.relu = nn.ReLU(False)

        for m in [self.conv]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.last_relu:
            x = self.relu(x)
        return x


class ConvBn3x3Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, padding=1, last_relu=True):
        super(ConvBn3x3Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                    dilation=dilation, padding=dilation if dilation > 1 else padding)
        self.bn = LearnedBatchNorm2d(out_channels)
        self.last_relu = last_relu
        if self.last_relu:
            self.relu = nn.ReLU(False)

        for m in [self.conv, self.bn]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
            else:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
                nn.init.constant_(m.running_var, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.last_relu:
            x = self.relu(x)
        return x


################################ Deconv2 Block ###################################
class Deconv2Block(nn.Module):
    def __init__(self, in_channels, out_channels, last_relu=True):
        super(Deconv2Block, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.last_relu = last_relu
        if self.last_relu:
            self.relu = nn.ReLU(False)

        for m in [self.deconv]:
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.deconv(x)
        if self.last_relu:
            x = self.relu(x)
        return x


class Deconv2BnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, last_relu=True):
        super(Deconv2BnBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.bn = LearnedBatchNorm2d(out_channels)
        self.last_relu = last_relu
        if self.last_relu:
            self.relu = nn.ReLU(False)

        for m in [self.deconv, self.bn]:
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
            else:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
                nn.init.constant_(m.running_var, 1)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        if self.last_relu:
            x = self.relu(x)
        return x




############################# mobilenet v2 core module WITH BN #########################
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, percent=1.0, input_ori=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup
        if not input_ori is None:
            expansion_dim = int(input_ori * expand_ratio)
        else:
            expansion_dim = int(inp * expand_ratio)
        expansion_dim = int(expansion_dim * percent)

        self.conv1 = nn.Conv2d(inp, expansion_dim, 1, 1, 0, bias=False)
        self.bn1 = BatchNorm2d(expansion_dim)
        self.relu1 = nn.ReLU6(False)

        self.conv2 = nn.Conv2d(expansion_dim, expansion_dim, 3, stride, 1, groups=expansion_dim, bias=False)
        self.bn2 = BatchNorm2d(expansion_dim)
        self.relu2 = nn.ReLU6(False)

        self.conv3 = nn.Conv2d(expansion_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = BatchNorm2d(oup)


    def forward(self, fea):
        x = self.conv1(fea)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.use_res_connect:
            x += fea

        return x


class InvertedResidual_NOBN(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, percent=1.0, input_ori=None):
        super(InvertedResidual_NOBN, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup
        if not input_ori is None:
            expansion_dim = int(input_ori * expand_ratio)
        else:
            expansion_dim = int(inp * expand_ratio)
        expansion_dim = int(expansion_dim * percent)

        self.conv1 = nn.Conv2d(inp, expansion_dim, 1, 1, 0, bias=True)
        self.relu1 = nn.ReLU6(False)

        self.conv2 = nn.Conv2d(expansion_dim, expansion_dim, 3, stride, 1, groups=expansion_dim, bias=True)
        self.relu2 = nn.ReLU6(False)

        self.conv3 = nn.Conv2d(expansion_dim, oup, 1, 1, 0, bias=True)


    def forward(self, fea):
        x = self.conv1(fea)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)

        if self.use_res_connect:
            x += fea

        return x

