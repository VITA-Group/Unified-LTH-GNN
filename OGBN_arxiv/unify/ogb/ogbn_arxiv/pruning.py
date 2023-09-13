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


def save_all(model, rewind_weight, optimizer, imp_num, epoch, save_path, save_name='default'):
    
    state = {
            'imp_num': imp_num,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'rewind_weight_mask': rewind_weight,
            'optimizer_state_dict': optimizer.state_dict()
        }

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Directory ", save_path, " is created.")

    filename = '{}/{}.pth'.format(save_path, save_name)
    torch.save(state, filename)


def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()

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


def add_mask(model, num_layers):

    for i in range(num_layers):
        mask_train = nn.Parameter(torch.ones_like(model.gcns[i].mlp[0].weight))
        mask_fixed = nn.Parameter(torch.ones_like(model.gcns[i].mlp[0].weight), requires_grad=False)
        AddTrainableMask.apply(model.gcns[i].mlp[0], 'weight', mask_train, mask_fixed)


def subgradient_update_mask(model, args):

    model.edge_mask1_train.grad.data.add_(args.s1 * torch.sign(model.edge_mask1_train.data))
    for i in range(args.num_layers):
        model.gcns[i].mlp[0].weight_mask_train.grad.data.add_(args.s2 * torch.sign(model.gcns[i].mlp[0].weight_mask_train.data))


def get_soft_mask_distribution(model):

    adj_mask_vector = model.edge_mask1_train.flatten()
    nonzero = torch.abs(adj_mask_vector) > 0
    adj_mask_vector = adj_mask_vector[nonzero]

    weight_mask_vector = torch.tensor([]).to(torch.device("cuda:0"))
    for i in range(28):
        weight_mask = model.gcns[i].mlp[0].weight_mask_train.flatten()
        nonzero = torch.abs(weight_mask) > 0
        weight_mask = weight_mask[nonzero]
        weight_mask_vector = torch.cat((weight_mask_vector, weight_mask))
    
    return adj_mask_vector.detach().cpu(), weight_mask_vector.detach().cpu()


def get_each_mask(mask_weight_tensor, threshold):
    
    ones  = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor) 
    mask = torch.where(mask_weight_tensor.abs() > threshold, ones, zeros)
    return mask

##### pruning remain mask percent #######
def get_final_mask_epoch(model, rewind_weight, args):

    adj_percent = args.pruning_percent_adj
    wei_percent = args.pruning_percent_wei
    num_layers = args.num_layers

    adj_mask, wei_mask = get_soft_mask_distribution(model)
    #adj_mask.add_((2 * torch.rand(adj_mask.shape) - 1) * 1e-5)
    adj_total = adj_mask.shape[0] # 2484941
    wei_total = wei_mask.shape[0] # 458752
    ### sort
    adj_y, adj_i = torch.sort(adj_mask.abs())
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    adj_thre_index = int(adj_total * adj_percent)
    adj_thre = adj_y[adj_thre_index]

    wei_thre_index = int(wei_total * wei_percent)
    wei_thre = wei_y[wei_thre_index]
    ### create mask dict 
    ori_edge_mask = model.edge_mask1_train.detach().cpu()
    rewind_weight['edge_mask1_train'] = get_each_mask(ori_edge_mask, adj_thre)
    rewind_weight['edge_mask2_fixed'] = rewind_weight['edge_mask1_train']

    adj_spar = rewind_weight['edge_mask2_fixed'].sum() * 100 / model.edge_num
    wei_all = 0
    wei_nonzero = 0

    for i in range(num_layers):
        key_train = 'gcns.{}.mlp.0.weight_mask_train'.format(i)
        key_fixed = 'gcns.{}.mlp.0.weight_mask_fixed'.format(i)
        rewind_weight[key_train] = get_each_mask(model.gcns[i].mlp[0].state_dict()['weight_mask_train'], wei_thre)
        rewind_weight[key_fixed] = rewind_weight[key_train]
        wei_nonzero += rewind_weight[key_fixed].sum()
        wei_all += rewind_weight[key_fixed].numel()
    wei_spar = wei_nonzero * 100 / wei_all
    return rewind_weight, adj_spar, wei_spar


def random_pruning(model, args, adj_percent, wei_percent):

    if adj_percent == 0 and wei_percent == 0:
        return

    num_layers = args.num_layers
    model.edge_mask1_train.requires_grad = False
    adj_total = model.edge_mask1_train.numel()
    adj_pruned_num = int(adj_total * adj_percent)
    adj_nonzero = model.edge_mask1_train.nonzero()
    adj_pruned_index = random.sample([i for i in range(adj_total)], adj_pruned_num)
    adj_pruned_list = adj_nonzero[adj_pruned_index].tolist()
    
    for i, j in adj_pruned_list:
        model.edge_mask1_train[i][j] = 0
        model.edge_mask2_fixed[i][j] = 0
    
    model.edge_mask1_train.requires_grad = True
    
    for i in range(num_layers):
        
        model.gcns[i].mlp[0].weight_mask_train.requires_grad = False
        wei_total = model.gcns[i].mlp[0].weight_mask_train.numel()
        wei_pruned_num = int(wei_total * wei_percent)
        wei_nonzero = model.gcns[i].mlp[0].weight_mask_train.nonzero()
        wei_pruned_index = random.sample([j for j in range(wei_total)], wei_pruned_num)
        wei_pruned_list = wei_nonzero[wei_pruned_index].tolist()

        for ii, (ai, wj) in enumerate(wei_pruned_list):
            model.gcns[i].mlp[0].weight_mask_train[ai][wj] = 0
            model.gcns[i].mlp[0].weight_mask_fixed[ai][wj] = 0

        model.gcns[i].mlp[0].weight_mask_train.requires_grad = True 


def print_sparsity(model, args):

    num_layers = args.num_layers
    adj_nonzero = model.edge_num
    adj_mask_nonzero = model.edge_mask2_fixed.sum().item()
    adj_spar = adj_mask_nonzero * 100 / adj_nonzero

    weight_total = 0
    weight_nonzero = 0

    for i in range(num_layers):
        weight_total += model.gcns[i].mlp[0].weight_mask_fixed.numel()
        weight_nonzero += model.gcns[i].mlp[0].weight_mask_fixed.sum().item()
    
    wei_spar = weight_nonzero * 100 / weight_total

    print("-" * 100)
    print("Sparsity: Adj:[{:.2f}%] Wei:[{:.2f}%]".format(adj_spar, wei_spar))
    print("-" * 100)

    return adj_spar, wei_spar


def add_trainable_mask_noise(model, args, c=1e-5):

    model.edge_mask1_train.requires_grad = False
    rand = (2 * torch.rand(model.edge_mask1_train.shape) - 1) * c
    rand = rand.to(model.edge_mask1_train.device) 
    rand = rand * model.edge_mask1_train
    model.edge_mask1_train.add_(rand)
    model.edge_mask1_train.requires_grad = True

    for i in range(args.num_layers):
        model.gcns[i].mlp[0].weight_mask_train.requires_grad = False
        rand = (2 * torch.rand(model.gcns[i].mlp[0].weight_mask_train.shape) - 1) * c
        rand = rand.to(model.gcns[i].mlp[0].weight_mask_train.device) 
        rand = rand * model.gcns[i].mlp[0].weight_mask_train
        model.gcns[i].mlp[0].weight_mask_train.add_(rand)
        model.gcns[i].mlp[0].weight_mask_train.requires_grad = True