# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
from collections import OrderedDict
from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "Cls"
_C.MODEL.DATASET = 'med'
_C.MODEL.NUM_CLASSES = 95
_C.MODEL.HIDDEN_DIM = 256
_C.MODEL.USE_PRETRAIN = True
_C.MODEL.USE_CLUSTER = False
_C.MODEL.USE_INIT = True
_C.MODEL.PRETRAIN_MODEL_DIR = ''
_C.MODEL.DATA_MEAN = []
_C.MODEL.DATA_STD = []

## TRANSFER setup
_C.MODEL.TRANSFER = CN()
_C.MODEL.TRANSFER.IS_OPEN = False
_C.MODEL.TRANSFER.MODEL_PREFIX = ''
_C.MODEL.TRANSFER.OUTPUT_NAMES = ('backbone.embed_bn', )
_C.MODEL.TRANSFER.WEIGHT_VALUES = (0.1, )
_C.MODEL.TRANSFER.METHOD = "hidden" # "logits"

## prunning setup
_C.MODEL.CUT = CN()
_C.MODEL.CUT.IS_OPEN = False
_C.MODEL.CUT.CUSTOM_OPEN = False
_C.MODEL.CUT.CURT_LAYERS = []
_C.MODEL.CUT.NEXT_LAYERS = []
_C.MODEL.CUT.CUT_PERCENTS = []
_C.MODEL.CUT.CUT_NAMES = []
_C.MODEL.CUT.CUT_ITERS = 100
_C.MODEL.CUT.MODEL_PREFIX = ''

_C.MODEL.CUT.MOBILENETV2 = CN()
_C.MODEL.CUT.MOBILENETV2.STAGE_BLOCKS = [1, 2, 3, 4, 3, 3, 1]
_C.MODEL.CUT.MOBILENETV2.DEFAULT_CUT_PERCENTS = [0.1, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5]
_C.MODEL.CUT.MOBILENETV2.CUSTOM_CUT_PERCENTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

_C.MODEL.RESNET50 = CN()
_C.MODEL.RESNET50.STAGE_BLOCKS = [3, 4, 6, 3]
_C.MODEL.RESNET50.DEFAULT_CUT_PERCENTS = (0.1, 0.3, 0.5, 0.7)
_C.MODEL.RESNET50.CUSTOM_CUT_PERCENTS = (0.0, 0.0, 0.0, 0.0)

_C.MODEL.MOBILENETV2 = CN()
_C.MODEL.MOBILENETV2.STAGE_BLOCKS = [1, 2, 3, 4, 3, 3, 1]
_C.MODEL.MOBILENETV2.DEFAULT_CUT_PERCENTS = [0.1, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5]
_C.MODEL.MOBILENETV2.CUSTOM_CUT_PERCENTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ("catalog://ImageNetPretrained/FAIR/20171220/X-101-32x8d", \
	               "catalog://ImageNetPretrained/MSRA/R-101", \
	               "catalog://ImageNetPretrained/MSRA/R-50", \
	               "./mobilenetv2_718.pth")


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = 224  # (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 224
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 224
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 224


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.CONV_BODY = "R-50" # R-101, MobileNetV2

_C.MODEL.MOBILENETV2 = CN()
_C.MODEL.MOBILENETV2.OUT_FEATURES = (0, 1, 1, 0, 1, 0, 1)
# Options: "FrozenBN", "GN", "SyncBN", "BN"
_C.MODEL.MOBILENETV2.NORM = "BN"
_C.MODEL.MOBILENETV2.STEM_OUT_CHANNELS = 32
_C.MODEL.MOBILENETV2.WIDTH_MULT = 1.0

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.BASE_LR = 0.1
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.STEPS = (80000, 140000, 180000)
_C.SOLVER.MAX_ITER = 200100
_C.SOLVER.CHECKPOINT_PERIOD = 5000

_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.IMS_PER_BATCH = 256


# ---------------------------------------------------------------------------- #
# TRANSFER Solver
# ---------------------------------------------------------------------------- #
_C.TRANSFER_SOLVER = CN()

_C.TRANSFER_SOLVER.BASE_LR = 0.1
_C.TRANSFER_SOLVER.BIAS_LR_FACTOR = 2

_C.TRANSFER_SOLVER.STEPS = (60000, 100000, 130000)
_C.TRANSFER_SOLVER.MAX_ITER = 200100
_C.TRANSFER_SOLVER.CHECKPOINT_PERIOD = 5000

_C.TRANSFER_SOLVER.WARMUP_FACTOR = 0.1
_C.TRANSFER_SOLVER.WARMUP_ITERS = 1000
_C.TRANSFER_SOLVER.WARMUP_METHOD = "linear"

_C.TRANSFER_SOLVER.MOMENTUM = 0.9
_C.TRANSFER_SOLVER.GAMMA = 0.1
_C.TRANSFER_SOLVER.WEIGHT_DECAY = 0.0005
_C.TRANSFER_SOLVER.WEIGHT_DECAY_BIAS = 0
_C.TRANSFER_SOLVER.IMS_PER_BATCH = 256


# ---------------------------------------------------------------------------- #
# CUT Solver
# ---------------------------------------------------------------------------- #
_C.CUT_SOLVER = CN()

_C.CUT_SOLVER.BASE_LR = 0.1
_C.CUT_SOLVER.BIAS_LR_FACTOR = 2

_C.CUT_SOLVER.STEPS = (20000,)
_C.CUT_SOLVER.MAX_ITER = 30100
_C.CUT_SOLVER.CHECKPOINT_PERIOD = 5000

_C.CUT_SOLVER.WARMUP_FACTOR = 0.1
_C.CUT_SOLVER.WARMUP_ITERS = 1000
_C.CUT_SOLVER.WARMUP_METHOD = "linear"

_C.CUT_SOLVER.MOMENTUM = 0.9
_C.CUT_SOLVER.GAMMA = 0.1
_C.CUT_SOLVER.WEIGHT_DECAY = 0.0005
_C.CUT_SOLVER.WEIGHT_DECAY_BIAS = 0
_C.CUT_SOLVER.IMS_PER_BATCH = 256


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 1


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_ROOT_DIR = ""
_C.OUTPUT_DIR = ""
_C.CUSTOM_OUTPUT_DIR = ""
_C.CUSTOM_ADD_NAME = ""

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

