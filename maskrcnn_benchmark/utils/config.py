import os
import pdb
import os.path as osp
from collections import OrderedDict
from maskrcnn_benchmark.utils.comm import get_world_size
from collections import OrderedDict


########################### common setup for both nets ########################
def setup_dataset(cfg, arch, dataset):
    assert(arch == 'Cls'), 'non-supported arch %s' % arch

    # possible datasets for each task
    cfg.DATASETS.TRAIN = ("%s_train"%dataset,)
    cfg.DATASETS.TEST = ("%s_val"%dataset,)
    num_classes = 95
    cfg.MODEL.NUM_CLASSES = num_classes


def setup_other(cfg):
    cfg.INPUT.MIN_SIZE_TEST = cfg.INPUT.MIN_SIZE_TRAIN
    cfg.INPUT.MAX_SIZE_TEST = cfg.INPUT.MAX_SIZE_TRAIN
    num_gpus = get_world_size()
    cfg.TEST.IMS_PER_BATCH = num_gpus
    cfg.MODEL.PRETRAIN_MODEL_DIR = cfg.OUTPUT_ROOT_DIR


def setup_transfer(cfg):
    if cfg.MODEL.TRANSFER.IS_OPEN:
        num_gpus = get_world_size()
        if num_gpus > 1:
            cfg.MODEL.TRANSFER.MODEL_PREFIX = 'module.'
        else:
            cfg.MODEL.TRANSFER.MODEL_PREFIX = ''

    cfg.SOLVER.BASE_LR = cfg.TRANSFER_SOLVER.BASE_LR
    cfg.SOLVER.WARMUP_FACTOR = cfg.TRANSFER_SOLVER.WARMUP_FACTOR
    cfg.SOLVER.WARMUP_ITERS = cfg.TRANSFER_SOLVER.WARMUP_ITERS
    cfg.SOLVER.STEPS = cfg.TRANSFER_SOLVER.STEPS
    cfg.SOLVER.MAX_ITER = cfg.TRANSFER_SOLVER.MAX_ITER
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TRANSFER_SOLVER.CHECKPOINT_PERIOD

def setup_cut(cfg):
    num_gpus = get_world_size()
    if num_gpus > 1:
        cfg.MODEL.CUT.MODEL_PREFIX = 'module.'
    else:
        cfg.MODEL.CUT.MODEL_PREFIX = ''
   
    if cfg.MODEL.CUT.IS_OPEN:
        curt_layers = []
        next_layers = []
        cut_percents = []
        cut_names = []

        body = cfg.MODEL.BACKBONE.CONV_BODY
        stage_blocks = cfg.MODEL.CUT.MOBILENETV2.STAGE_BLOCKS
        stage_percents = cfg.MODEL.CUT.MOBILENETV2.DEFAULT_CUT_PERCENTS
        if cfg.MODEL.CUT.CUSTOM_OPEN:
            custom_percents = cfg.MODEL.CUT.MOBILENETV2.CUSTOM_CUT_PERCENTS
            assert(len(custom_percents) == len(stage_percents)), 'the length of custom and default percents must match !'
            stage_percents = custom_percents

        num_blocks = sum(stage_blocks)
        stage_percents_v2 = []
        for num, percent in zip(stage_blocks, stage_percents):
            for _ in range(num):
                stage_percents_v2.append(percent)
        assert(num_blocks == len(stage_percents_v2))
        stage_percents = stage_percents_v2

        prefix = '%sbackbone' % (cfg.MODEL.CUT.MODEL_PREFIX)
        block_name0 = '%s.features.0' % prefix
        curt_name   = '%s.0' % block_name0
        block_name1 = '%s.features.1' % prefix
        next_name   = '%s.conv.1' % block_name1
        curt_layers.append(curt_name)
        next_layers.append(next_name)
        cut_percents.append(stage_percents[0])
        cut_names.append(block_name0)

        for stage_id, stage_percent in enumerate(stage_percents[1:], 2):
            block_name = '%s.features.%d' % (prefix, stage_id)
            curt_name = '%s.conv.0.0' % block_name
            next_name = '%s.conv.2' % block_name
            curt_layers.append(curt_name)
            next_layers.append(next_name)
            cut_percents.append(stage_percent)
            cut_names.append(block_name)

        cfg.MODEL.CUT.CURT_LAYERS = curt_layers
        cfg.MODEL.CUT.NEXT_LAYERS = next_layers
        cfg.MODEL.CUT.CUT_PERCENTS = cut_percents
        cfg.MODEL.CUT.CUT_NAMES = cut_names

    cfg.SOLVER.BASE_LR = cfg.TRANSFER_SOLVER.BASE_LR
    cfg.SOLVER.WARMUP_FACTOR = cfg.TRANSFER_SOLVER.WARMUP_FACTOR
    cfg.SOLVER.WARMUP_ITERS = cfg.TRANSFER_SOLVER.WARMUP_ITERS
    cfg.SOLVER.STEPS = cfg.TRANSFER_SOLVER.STEPS
    cfg.SOLVER.MAX_ITER = cfg.TRANSFER_SOLVER.MAX_ITER
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TRANSFER_SOLVER.CHECKPOINT_PERIOD


def setup_output_name(cfg, dict_list, out_name):
    for name_dict in dict_list:
        for key, val in name_dict.items():
            if val > 0:
                if isinstance(val, float):
                    out_name += '%s%.1f_' % (key, val)
                elif isinstance(val, str):
                    out_name += '%s%s_' % (key, val)
                else:
                    out_name += '%s%d_' % (key, val)
    out_name = out_name[:-1]

    if cfg.CUSTOM_OUTPUT_DIR == "":
        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_ROOT_DIR, out_name)
    else:
        cfg.OUTPUT_DIR = cfg.CUSTOM_OUTPUT_DIR

    if not cfg.CUSTOM_ADD_NAME == '':
        cfg.OUTPUT_DIR += '/' + cfg.CUSTOM_ADD_NAME


################################ MAIN CONTROL ############################
def integrate_cfg(cfg):
    # architecture
    arch = cfg.MODEL.META_ARCHITECTURE
    dataset = cfg.MODEL.DATASET
    body = cfg.MODEL.BACKBONE.CONV_BODY

    setup_dataset(cfg, arch, dataset)
    setup_other(cfg)
    setup_transfer(cfg)
    setup_cut(cfg)

    solver_dict = {}
    solver_dict['Iter'] = cfg.SOLVER.MAX_ITER
    solver_dict['Pretrain'] = cfg.MODEL.USE_PRETRAIN
    dict_list = [solver_dict]

    out_name = "%s_%s_" % (body, dataset)
    setup_output_name(cfg, dict_list, out_name)
    print("************************")
    print(cfg.OUTPUT_DIR)
    print("************************")

