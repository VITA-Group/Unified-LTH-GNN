import torch
import torch.nn as nn
from abc import ABC
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pdb
import torch.nn.init as init
import math
# gcns.ginlayers[0].apply_func.mlp.linear

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


class AddTrainableMask(ABC):

    _tensor_name: str
    
    def __init__(self):
        pass
    
    def __call__(self, module, inputs):

        setattr(module, self._tensor_name, self.apply_mask(module))

    def apply_mask(self, module):

        mask_train = getattr(module, self._tensor_name + "_mask_train")
        mask_fixed = getattr(module, self._tensor_name + "_mask_fixed")
        orig_weight = getattr(module, self._tensor_name + "_orig_weight")
        pruned_weight = mask_train * mask_fixed * orig_weight 
        
        return pruned_weight

    @classmethod
    def apply(cls, module, name, mask_train, mask_fixed, *args, **kwargs):

        method = cls(*args, **kwargs)  
        method._tensor_name = name
        orig = getattr(module, name)

        module.register_parameter(name + "_mask_train", mask_train.to(dtype=orig.dtype))
        module.register_parameter(name + "_mask_fixed", mask_fixed.to(dtype=orig.dtype))
        module.register_parameter(name + "_orig_weight", orig)
        del module._parameters[name]

        setattr(module, name, method.apply_mask(module))
        module.register_forward_pre_hook(method)

        return method


def add_mask(model):
    # model.layers[0].heads[0].fc
    # model.layers[0].heads[0].attn_fc
    for layer in range(2):
        for head in range(8):
            mask1_train = nn.Parameter(torch.ones_like(model.layers[layer].heads[head].fc.weight))
            mask1_fixed = nn.Parameter(torch.ones_like(model.layers[layer].heads[head].fc.weight), requires_grad=False)
            mask2_train = nn.Parameter(torch.ones_like(model.layers[layer].heads[head].attn_fc.weight))
            mask2_fixed = nn.Parameter(torch.ones_like(model.layers[layer].heads[head].attn_fc.weight), requires_grad=False)
            AddTrainableMask.apply(model.layers[layer].heads[head].fc, 'weight', mask1_train, mask1_fixed)
            AddTrainableMask.apply(model.layers[layer].heads[head].attn_fc, 'weight', mask2_train, mask2_fixed)
            if layer == 1: break
 
 
def subgradient_update_mask(model, args):

    model.adj_mask1_train.grad.data.add_(args['s1'] * torch.sign(model.adj_mask1_train.data))
    for layer in range(2):
        for head in range(8):
            model.layers[layer].heads[head].fc.weight_mask_train.grad.data.add_(args['s2'] * torch.sign(model.layers[layer].heads[head].fc.weight_mask_train.data))
            model.layers[layer].heads[head].attn_fc.weight_mask_train.grad.data.add_(args['s2'] * torch.sign(model.layers[layer].heads[head].attn_fc.weight_mask_train.data))
            if layer == 1: break


def get_mask_distribution(model):

    adj_mask_tensor = model.adj_mask1_train.flatten()
    nonzero = torch.abs(adj_mask_tensor) > 0
    adj_mask_tensor = adj_mask_tensor[nonzero] # 13264 - 2708

    weight_mask_vector = torch.tensor([]).to(torch.device("cuda:0"))
    for layer in range(2):
        for head in range(8):
            weight_mask1 = model.layers[layer].heads[head].fc.weight_mask_train.flatten()
            nonzero = torch.abs(weight_mask1) > 0
            weight_mask1 = weight_mask1[nonzero]

            weight_mask2 = model.layers[layer].heads[head].attn_fc.weight_mask_train.flatten()
            nonzero = torch.abs(weight_mask2) > 0
            weight_mask2 = weight_mask2[nonzero]
            weight_mask_vector = torch.cat((weight_mask_vector, weight_mask1))
            weight_mask_vector = torch.cat((weight_mask_vector, weight_mask2))
            if layer == 1: break

    return adj_mask_tensor.detach().cpu(), weight_mask_vector.detach().cpu()


def get_each_mask(mask_weight_tensor, threshold):
    
    ones  = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor) 
    mask = torch.where(mask_weight_tensor.abs() > threshold, ones, zeros)
    return mask

def get_each_mask_admm(mask_weight_tensor, threshold):
    
    zeros = torch.zeros_like(mask_weight_tensor) 
    mask = torch.where(mask_weight_tensor.abs() > threshold, mask_weight_tensor, zeros)
    return mask

##### pruning remain mask percent #######
def get_final_mask_epoch(model, rewind_weight, args):
    
    adj_percent=args['pruning_percent_adj']
    wei_percent=args['pruning_percent_wei']

    adj_mask, wei_mask = get_mask_distribution(model)

    adj_total = adj_mask.shape[0]
    wei_total = wei_mask.shape[0]
    ### sort
    adj_y, adj_i = torch.sort(adj_mask.abs())
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    adj_thre_index = int(adj_total * adj_percent)
    adj_thre = adj_y[adj_thre_index]
    
    wei_thre_index = int(wei_total * wei_percent)
    wei_thre = wei_y[wei_thre_index]

    mask_dict = {}
    ori_adj_mask = model.adj_mask1_train.detach().cpu()

    rewind_weight['adj_mask1_train'] = get_each_mask(ori_adj_mask, adj_thre)
    rewind_weight['adj_mask2_fixed'] = get_each_mask(ori_adj_mask, adj_thre)
    
    adj_all = rewind_weight['adj_mask2_fixed'].numel()
    adj_nozero = rewind_weight['adj_mask2_fixed'].sum()
    adj_spar = adj_nozero * 100 / adj_all

    wei_all = 0
    wei_nonzero = 0

    for layer in range(2):
        for head in range(8):

            key_train1 = 'layers.{}.heads.{}.fc.weight_mask_train'.format(layer, head)
            key_fixed1 = 'layers.{}.heads.{}.fc.weight_mask_fixed'.format(layer, head)

            key_train2 = 'layers.{}.heads.{}.attn_fc.weight_mask_train'.format(layer, head)
            key_fixed2 = 'layers.{}.heads.{}.attn_fc.weight_mask_fixed'.format(layer, head)

            rewind_weight[key_train1] = get_each_mask(model.layers[layer].heads[head].fc.state_dict()['weight_mask_train'], wei_thre)
            rewind_weight[key_fixed1] = rewind_weight[key_train1]

            rewind_weight[key_train2] = get_each_mask(model.layers[layer].heads[head].attn_fc.state_dict()['weight_mask_train'], wei_thre)
            rewind_weight[key_fixed2] = rewind_weight[key_train2] 

            wei_all += rewind_weight[key_fixed1].numel()
            wei_all += rewind_weight[key_fixed2].numel()
            wei_nonzero += rewind_weight[key_fixed1].sum()
            wei_nonzero += rewind_weight[key_fixed2].sum()
            if layer == 1: break

    wei_spar = wei_nonzero * 100 / wei_all
    return rewind_weight, adj_spar, wei_spar





##### random pruning #######
def random_pruning(model, adj_percent, wei_percent):

    model.adj_mask1_train.requires_grad = False
    adj_nonzero = model.adj_mask1_train.nonzero()
    adj_total = adj_nonzero.shape[0]
    adj_pruned_num = int(adj_total * adj_percent)
    adj_index = random.sample([i for i in range(adj_total)], adj_pruned_num)

    adj_pruned = adj_nonzero[adj_index].tolist()
    for i, j in adj_pruned:
        model.adj_mask1_train[i][j] = 0
        model.adj_mask2_fixed[i][j] = 0
    model.adj_mask1_train.requires_grad = True


    for layer in range(2):
        for head in range(8):

            model.layers[layer].heads[head].fc.weight_mask_train.requires_grad = False
            model.layers[layer].heads[head].attn_fc.weight_mask_train.requires_grad = False

            wei1_nonzero = model.layers[layer].heads[head].fc.weight_mask_train.nonzero()
            wei2_nonzero = model.layers[layer].heads[head].attn_fc.weight_mask_train.nonzero()

            wei1_total = wei1_nonzero.shape[0]
            wei2_total = wei2_nonzero.shape[0]

            wei1_pruned_num = int(wei1_total * wei_percent)
            wei2_pruned_num = int(wei2_total * wei_percent)

            wei1_index = random.sample([i for i in range(wei1_total)], wei1_pruned_num)
            wei2_index = random.sample([i for i in range(wei2_total)], wei2_pruned_num)

            wei1_pruned = wei1_nonzero[wei1_index].tolist()
            wei2_pruned = wei2_nonzero[wei2_index].tolist()

            for i, j in wei1_pruned:
                model.layers[layer].heads[head].fc.weight_mask_train[i][j] = 0
                model.layers[layer].heads[head].fc.weight_mask_fixed[i][j] = 0
    
            for i, j in wei2_pruned:
                model.layers[layer].heads[head].attn_fc.weight_mask_train[i][j] = 0
                model.layers[layer].heads[head].attn_fc.weight_mask_fixed[i][j] = 0

            model.layers[layer].heads[head].fc.weight_mask_train.requires_grad = True
            model.layers[layer].heads[head].attn_fc.weight_mask_train.requires_grad = True
            if layer == 1: break



def print_sparsity(model):

    adj_nonzero = model.edge_num
    adj_mask_nonzero = model.adj_mask2_fixed.sum().item()
    adj_spar = adj_mask_nonzero * 100 / adj_nonzero
    
    weight_total = 0
    weight_nonzero = 0
    for layer in range(2):
        for head in range(8):
            weight_total += model.layers[layer].heads[head].fc.weight_mask_fixed.numel()
            weight_total += model.layers[layer].heads[head].attn_fc.weight_mask_fixed.numel()
            weight_nonzero += model.layers[layer].heads[head].fc.weight_mask_fixed.sum().item()
            weight_nonzero += model.layers[layer].heads[head].attn_fc.weight_mask_fixed.sum().item()
            if layer == 1: break

    wei_spar = weight_nonzero * 100 / weight_total
    print("-" * 100)
    print("Sparsity: Adj:[{:.2f}%] Wei:[{:.2f}%]".format(adj_spar, wei_spar))
    print("-" * 100)
    return adj_spar, wei_spar


def tensor_add_noise(in_tensor, c):

    noise_tensor = (2 * torch.rand(in_tensor.shape) - 1) * c
    noise_tensor = noise_tensor.to(in_tensor.device)
    noise_tensor = noise_tensor * in_tensor
    in_tensor.add_(noise_tensor)
    

def add_trainable_mask_noise(model, c=1e-5):
    
    
    model.adj_mask1_train.requires_grad = False
    tensor_add_noise(model.adj_mask1_train, c)
    model.adj_mask1_train.requires_grad = True
    for layer in range(2):
        for head in range(8):
            model.layers[layer].heads[head].fc.weight_mask_train.requires_grad = False
            model.layers[layer].heads[head].attn_fc.weight_mask_train.requires_grad = False
            tensor_add_noise(model.layers[layer].heads[head].fc.weight_mask_train, c)
            tensor_add_noise(model.layers[layer].heads[head].attn_fc.weight_mask_train, c)
            model.layers[layer].heads[head].fc.weight_mask_train.requires_grad = True
            model.layers[layer].heads[head].attn_fc.weight_mask_train.requires_grad = True
            if layer == 1: break