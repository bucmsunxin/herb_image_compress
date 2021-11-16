import numpy as np
import logging
from tqdm import tqdm
import torch.nn.functional as F
import time
import torch
import torch.nn as nn
import pdb
import os
from ..utils.comm import synchronize
from ..utils.comm import all_gather
from ..utils.comm import is_main_process

logger = logging.getLogger(__name__)

def cut_load_model(model, curt_layer, next_layer, cut_percent, remove_bn=False):
    # get param layers
    layer_prefix = 'backbone.features'
    layer_idx = int(curt_layer.split('.')[2])
    if layer_idx == 0:
        curt_conv_module_name = '%s.%d.0' % (layer_prefix, layer_idx)
        curt_bn_module_name = '%s.%d.1' % (layer_prefix, layer_idx)
        dw_conv_module_name = '%s.%d.conv.0.0' % (layer_prefix, layer_idx + 1)
        dw_bn_module_name = '%s.%d.conv.0.1' % (layer_prefix, layer_idx + 1)
        next_conv_module_name = '%s.%d.conv.1' % (layer_prefix, layer_idx + 1)
    else:
        curt_conv_module_name = '%s.%d.conv.0.0' % (layer_prefix, layer_idx)
        curt_bn_module_name = '%s.%d.conv.0.1' % (layer_prefix, layer_idx)
        dw_conv_module_name = '%s.%d.conv.1.0' % (layer_prefix, layer_idx)
        dw_bn_module_name = '%s.%d.conv.1.1' % (layer_prefix, layer_idx)
        next_conv_module_name = '%s.%d.conv.2' % (layer_prefix, layer_idx)  

    curt_conv_weight_name = '%s.weight' % curt_conv_module_name
    curt_conv_bias_name = '%s.bias' % curt_conv_module_name
    curt_conv_has_bias = True if curt_conv_bias_name in model.state_dict().keys() else False
    curt_bn_weight_name = '%s.weight' % curt_bn_module_name
    curt_bn_bias_name = '%s.bias' % curt_bn_module_name

    dw_conv_weight_name = '%s.weight' % dw_conv_module_name
    dw_conv_bias_name = '%s.bias' % dw_conv_module_name
    dw_conv_has_bias = True if dw_conv_bias_name in model.state_dict().keys() else False
    dw_bn_weight_name = '%s.weight' % dw_bn_module_name
    dw_bn_bias_name = '%s.bias' % dw_bn_module_name

    next_conv_weight_name = '%s.weight' % next_conv_module_name
    next_conv_bias_name = '%s.bias' % next_conv_module_name
    next_conv_has_bias = True if next_conv_bias_name in model.state_dict().keys() else False

    # get preserved dim
    middle_dim = None
    for name, module in model.named_modules():
        if name == curt_conv_module_name:
            middle_dim = module.weight.shape[0] - int(module.weight.shape[0] * cut_percent)
            break

    ##### add temp selected_indx #####
    selected_indx = np.arange(middle_dim)

    # create new module for the cutd model
    assert(middle_dim is not None)
    for name, module in model.named_modules():
        ## current layers
        if name == curt_conv_module_name:
            module.out_channels = middle_dim
            module.weight = nn.Parameter(module.weight[selected_indx,:,:,:])
            if curt_conv_has_bias:
                module.bias = nn.Parameter(module.bias[selected_indx])

        if name == curt_bn_module_name:
            module.num_features = middle_dim
            module.weight = nn.Parameter(module.weight[selected_indx])
            module.bias = nn.Parameter(module.bias[selected_indx])
            module.running_mean = module.running_mean[selected_indx]
            module.running_var = module.running_var[selected_indx]

        ## dw conv layers
        if name == dw_conv_module_name:
            module.out_channels = middle_dim
            module.in_channels = middle_dim
            module.groups = middle_dim
            module.weight = nn.Parameter(module.weight[selected_indx,:,:,:])
            if dw_conv_has_bias:
                module.bias = nn.Parameter(module.bias[selected_indx])

        if name == dw_bn_module_name:
            module.num_features = middle_dim
            module.weight = nn.Parameter(module.weight[selected_indx])
            module.bias = nn.Parameter(module.bias[selected_indx])
            module.running_mean = module.running_mean[selected_indx]
            module.running_var = module.running_var[selected_indx]

        ## next layers
        if name == next_conv_module_name:
            module.in_channels = middle_dim
            module.weight = nn.Parameter(module.weight[:,selected_indx,:,:])
    return model


def compute_exclusive_error(curt_map, next_map, next_conv, weight, bias=None):
    in_channels = next_conv.in_channels
    out_channels = next_conv.out_channels
    stride = next_conv.stride
    padding = next_conv.padding
    has_bias = next_conv.bias

    if has_bias:
        assert(bias is not None)
        tmp_next_map = F.conv2d(input=curt_map, weight=weight, bias=bias, stride=stride, padding=padding)
    else:
        assert(bias is None)
        tmp_next_map = F.conv2d(input=curt_map, weight=weight, bias=None, stride=stride, padding=padding)

    tmp_error = (next_map - tmp_next_map) ** 2
    tmp_error = torch.sum(tmp_error, dim=(1,2,3))
    tmp_error = tmp_error.to('cpu').numpy()[0]
    return tmp_error

def extract_exclusive_map_channel(data, delete_channel):
    tmp_data = data[:,1:,...].clone()
    if delete_channel == 0:
        tmp_data = data[:,1:,...]
    elif delete_channel == data.shape[1] -1:
        tmp_data = data[:,:delete_channel,...]
    else:
        tmp_data[:,0:delete_channel,...] = data[:,0:delete_channel,...]
        tmp_data[:,delete_channel:,...] = data[:,delete_channel+1:,...]
    return  tmp_data

def extract_exclusive_weight_channel(data, delete_channel):
    return extract_exclusive_map_channel(data, delete_channel)

def extract_exclusive_bias_channel(data, delete_channel):
    tmp_data = data[1:].clone()
    if delete_channel == 0:
        tmp_data = data[1:]
    elif delete_channel == data.shape[0] -1:
        tmp_data = data[:delete_channel]
    else:
        tmp_data[:delete_channel] = data[:delete_channel]
        tmp_data[delete_channel:] = data[:,delete_channel+1:]
    return tmp_data    

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    synchronize()
    if not is_main_process():
        return

    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger.warning(
        "Number of images that were gathered from multiple processes is not "
        "a contiguous set. Some images might be missing from the evaluation"
        )
    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions



class Cutter(object):
    def __init__(self, model, data_loader, curt_layer, next_layer, cut_percent):
        self.model = model
        self.modules = dict(self.model.named_modules())
        self.data_loader = data_loader
        self.curt_layer = curt_layer
        self.next_layer = next_layer
        self.cut_percent = cut_percent
        self.layer_prefix = 'backbone.features'
        self.layer_idx = int(self.curt_layer.split('.')[2])

        self.curt_feature_dict = dict()
        self.next_feature_dict = dict()
        self.device = None
        self.selected_indx = None
        self.middle_dim = None

    def cut(self):
        self.get_cut_name()
        results_dict = self.get_cut_error()
        self.get_cut_index(results_dict)
        synchronize()
        self.get_cut_model()
        synchronize()
        return self.model

    def hook_forward(self, name):
        def hook(module, input, output):
            self.curt_feature_dict[name] =  input
            self.next_feature_dict[name] = output
        return hook

    def get_cut_name(self):
        if self.layer_idx == 0:
            self.curt_conv_module_name = '%s.%d.0' % (self.layer_prefix, self.layer_idx)
            self.curt_bn_module_name = '%s.%d.1' % (self.layer_prefix, self.layer_idx)
            self.dw_conv_module_name = '%s.%d.conv.0.0' % (self.layer_prefix, self.layer_idx + 1)
            self.dw_bn_module_name = '%s.%d.conv.0.1' % (self.layer_prefix, self.layer_idx + 1)
            self.next_conv_module_name = '%s.%d.conv.1' % (self.layer_prefix, self.layer_idx + 1)
        else:
            self.curt_conv_module_name = '%s.%d.conv.0.0' % (self.layer_prefix, self.layer_idx)
            self.curt_bn_module_name = '%s.%d.conv.0.1' % (self.layer_prefix, self.layer_idx)
            self.dw_conv_module_name = '%s.%d.conv.1.0' % (self.layer_prefix, self.layer_idx)
            self.dw_bn_module_name = '%s.%d.conv.1.1' % (self.layer_prefix, self.layer_idx)
            self.next_conv_module_name = '%s.%d.conv.2' % (self.layer_prefix, self.layer_idx)            

        self.curt_conv_weight_name = '%s.weight' % self.curt_conv_module_name
        self.curt_conv_bias_name = '%s.bias' % self.curt_conv_module_name
        self.curt_conv_has_bias = True if self.curt_conv_bias_name in self.model.state_dict().keys() else False
        self.curt_bn_weight_name = '%s.weight' % self.curt_bn_module_name
        self.curt_bn_bias_name = '%s.bias' % self.curt_bn_module_name

        self.dw_conv_weight_name = '%s.weight' % self.dw_conv_module_name
        self.dw_conv_bias_name = '%s.bias' % self.dw_conv_module_name
        self.dw_conv_has_bias = True if self.dw_conv_bias_name in self.model.state_dict().keys() else False
        self.dw_bn_weight_name = '%s.weight' % self.dw_bn_module_name
        self.dw_bn_bias_name = '%s.bias' % self.dw_bn_module_name

        self.next_conv_weight_name = '%s.weight' % self.next_conv_module_name
        self.next_conv_bias_name = '%s.bias' % self.next_conv_module_name
        self.next_conv_has_bias = True if self.next_conv_bias_name in self.model.state_dict().keys() else False


    def get_cut_error(self):
        module = self.modules[self.next_conv_module_name]
        handle_current = module.register_forward_hook(self.hook_forward(self.next_conv_module_name))
        logger.info('extring reference and channel features ...')
        results_dict = {}
        for i, batch in tqdm(enumerate(self.data_loader)):
            ## forwarding to get the input & output reference feature map
            images, targets, image_ids = batch
            assert (len(image_ids) == 1)
            images = images.to('cuda')
            with torch.no_grad():
                _ = self.model(images)

            ## get next conv weigth & bias
            next_weight = None
            next_bias = None
            for name, param in self.model.named_parameters():
                if name == self.next_conv_weight_name:
                    next_weight = param.data.clone()
                if self.next_conv_has_bias:
                    if name == self.next_conv_bias_name:
                        next_bias = param.data.clone()

            next_conv = None
            for name, module in self.model.named_modules():
                if name == self.next_conv_module_name:
                    next_conv = module

            ## compute L2 loss by removing each channel
            current_map = self.curt_feature_dict[self.next_conv_module_name][0] #input is a tuple
            next_map = self.next_feature_dict[self.next_conv_module_name]
            self.device = next_weight.device
            error_list = []
            
            for channel in range(current_map.shape[1]):
                tmp_map = extract_exclusive_map_channel(current_map, channel)
                tmp_weight = extract_exclusive_weight_channel(next_weight, channel)
                tmp_bias = None
                if self.next_conv_has_bias:
                    tmp_bias = extract_exclusive_bias_channel(next_bias, channel)
                tmp_error = compute_exclusive_error(tmp_map, next_map, next_conv, tmp_weight, bias=tmp_bias)
                error_list.append(tmp_error)
            results_dict.update({image_ids[0]: [error_list]})
        handle_current.remove()
        return results_dict


    def get_cut_index(self, results_dict):
        accu_results = _accumulate_predictions_from_multiple_gpus(results_dict)
        if not is_main_process():
            return
        L2_error = np.array(accu_results).squeeze()
        channel_error = np.sum(L2_error, axis=0)
        indx = np.argsort(channel_error)
        cut_num = int(L2_error.shape[1] * self.cut_percent)
        remain_num = L2_error.shape[1] - cut_num
        selected_indx = indx[cut_num:]
        selected_indx = torch.from_numpy(selected_indx).to(self.device)
        self.middle_dim = selected_indx.numel()
        logger.info('channel from %d to %d'%(L2_error.shape[1], remain_num))
        torch.save(selected_indx, 'selected_indx.pth')


    def get_cut_model(self):
        synchronize()
        selected_indx = torch.load('selected_indx.pth')
        if is_main_process():
            os.system('rm -rf ./selected_indx.pth')
        synchronize()

        assert(self.middle_dim is not None)
        for name, module in self.model.named_modules():
            ## current layers
            if name == self.curt_conv_module_name:
                module.out_channels = self.middle_dim
                module.weight = nn.Parameter(module.weight[selected_indx,:,:,:])
                if self.curt_conv_has_bias:
                    module.bias = nn.Parameter(module.bias[selected_indx])

            if name == self.curt_bn_module_name:
                module.num_features = self.middle_dim
                module.weight = nn.Parameter(module.weight[selected_indx])
                module.bias = nn.Parameter(module.bias[selected_indx])
                module.running_mean = module.running_mean[selected_indx]
                module.running_var = module.running_var[selected_indx]
 
            ## dw conv layers
            if name == self.dw_conv_module_name:
                module.out_channels = self.middle_dim
                module.in_channels = self.middle_dim
                module.groups = self.middle_dim
                module.weight = nn.Parameter(module.weight[selected_indx,:,:,:])
                if self.dw_conv_has_bias:
                    module.bias = nn.Parameter(module.bias[selected_indx])

            if name == self.dw_bn_module_name:
                module.num_features = self.middle_dim
                module.weight = nn.Parameter(module.weight[selected_indx])
                module.bias = nn.Parameter(module.bias[selected_indx])
                module.running_mean = module.running_mean[selected_indx]
                module.running_var = module.running_var[selected_indx]
 
            ## next layers
            if name == self.next_conv_module_name:
                module.in_channels = self.middle_dim
                module.weight = nn.Parameter(module.weight[:,selected_indx,:,:])

