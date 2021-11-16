# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
from collections import OrderedDict
import torch
import pdb
import os
import os.path as osp

from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint


    def load_iter(self, count_iter, f=None):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file_iter(count_iter)
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""

        model_name = last_saved.split('/')[-1]
        last_saved = os.path.join(self.save_dir, model_name).strip()
        assert(osp.exists(last_saved))
        return last_saved

    def get_checkpoint_file_iter(self, count_iter):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""

        model_name = 'model_%07d.pth' % count_iter
        last_saved = os.path.join(self.save_dir, model_name).strip()
        assert(osp.exists(last_saved))
        return last_saved


    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # # catalog lookup
        # if f.startswith("catalog://"):
        #     paths_catalog = import_file(
        #         "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
        #     )
        #     catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
        #     self.logger.info("{} points to {}".format(f, catalog_f))
        #     f = catalog_f
        # # download url files
        # if f.startswith("http"):
        #     # if the file is a url path, download it and cache it
        #     cached_f = cache_url(f)
        #     self.logger.info("url {} cached in {}".format(f, cached_f))
        #     f = cached_f
        # # convert Caffe2 checkpoint from pkl
        # if f.endswith(".pkl"):
        #     return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded


def match_layer_names(cfg, model, pre_state_dict, prefix_old, prefix_new, stage_layer_replicas, head_prefix=None):
    # match blocks
    matched_blocks = OrderedDict()
    matched_blocks['%s.%d'%(prefix_old, 0)] = '%s.%s' % (prefix_new,'first_stem')
    num_layers = sum(stage_layer_replicas)
    assert(num_layers == 17)
    acc_cnt = 1
    for stage_idx, num_replica in enumerate(stage_layer_replicas, 1):
        for i in xrange(num_replica):
            old_name = '%s.%s' % (prefix_old, acc_cnt)
            new_name = '%s.stage%d_layer%d' % (prefix_new, stage_idx, i)
            matched_blocks[old_name] = new_name
            acc_cnt += 1
    matched_blocks['%s.%d'%(prefix_old, 18)] = '%s.%s' % (prefix_new,'last_stem')

    # match layer names in each block
    matched_layers = OrderedDict()

    # fisrt and last stem
    matched_layers['0.weight'] = 'conv1.weight'
    matched_layers['1.weight'] = 'bn1.weight'
    matched_layers['1.bias'] = 'bn1.bias'
    matched_layers['1.running_mean'] = 'bn1.running_mean'
    matched_layers['1.running_var'] = 'bn1.running_var'

    # inverse module
    matched_layers['conv.0.weight'] = 'conv1.weight'
    matched_layers['conv.1.weight'] = 'bn1.weight'
    matched_layers['conv.1.bias'] = 'bn1.bias'
    matched_layers['conv.1.running_mean'] = 'bn1.running_mean'
    matched_layers['conv.1.running_var'] = 'bn1.running_var'

    matched_layers['conv.3.weight'] = 'conv2.weight'
    matched_layers['conv.4.weight'] = 'bn2.weight'
    matched_layers['conv.4.bias'] = 'bn2.bias'
    matched_layers['conv.4.running_mean'] = 'bn2.running_mean'
    matched_layers['conv.4.running_var'] = 'bn2.running_var'

    matched_layers['conv.6.weight'] = 'conv3.weight'
    matched_layers['conv.7.weight'] = 'bn3.weight'
    matched_layers['conv.7.bias'] = 'bn3.bias'
    matched_layers['conv.7.running_mean'] = 'bn3.running_mean'
    matched_layers['conv.7.running_var'] = 'bn3.running_var'

    # match each layer
    new_state_dict = model.state_dict()
    new_keys = new_state_dict.keys()

    for old_key, val in pre_state_dict.items():
        if not prefix_old in old_key:
            print('ignoring pretrain layer %s' % old_key)
            continue

        if '%s.0'%prefix_old in old_key or '%s.18'%prefix_old in old_key:
            _, block, layer_idx, weight_type = old_key.split('.')
            old_block_name = '%s.%s' % (prefix_old, block)
            old_layer_name = '%s.%s' % (layer_idx, weight_type)
        else:
            _, block, layer_type, layer_idx, weight_type = old_key.split('.')
            old_block_name = '%s.%s' % (prefix_old, block)
            old_layer_name = '%s.%s.%s' % (layer_type, layer_idx, weight_type)

        new_block_name = matched_blocks[old_block_name]
        new_layer_name = matched_layers[old_layer_name]
        if head_prefix is not None:
            new_key = '%s.%s.%s' % (head_prefix, new_block_name, new_layer_name)
        else:
            new_key = '%s.%s' % (new_block_name, new_layer_name)

        if cfg.MODEL.MOBILENETV2.USE_EXPAND_1280 is False:
            if 'last_stem' in new_key:
                print('ignoring %s' % new_key)
                continue

        try:
            assert(new_key in new_keys)
            assert(new_state_dict[new_key].shape == val.shape)
        except:
            print('old: %s, new:%s' % (old_key, new_key))

        print('copying layer %s from layer %s' % (new_key, old_key))
        new_state_dict[new_key] = val

    return new_state_dict
