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
from tqdm import tqdm

def retrain_operation(dataset, model, all_ckpt):

    dataset.adj = all_ckpt['adj']
    rewind_weight = all_ckpt['rewind_weight_mask']
    mask_ckpt = all_ckpt['model_state_dict']
    
    for i in range(28):

        key_fixed1 = 'gcns.{}.mlp.0.weight_mask_fixed'.format(i)
        key_fixed2 = 'gcns.{}.mlp.3.weight_mask_fixed'.format(i)
        rewind_weight[key_fixed1] = mask_ckpt[key_fixed1]
        rewind_weight[key_fixed2] = mask_ckpt[key_fixed2]
        
    model.load_state_dict(rewind_weight)
        


def remove_mask(model):

    model.edge_mask = None
    for i in range(28):
        model.gcns[i].edge_mask = None


def pruning_real_adj_matrix(it, model, ogbndataset, edges_orig_idx, pruning_num):
    
    ### 1. get learnable mask
    mask = model.edge_mask.detach()
    ### 2. compute pruning index
    edge_num = mask.numel()
    mask_sorted, _ = torch.sort(mask.abs().flatten())
    # mask_thre_index = int(edge_num * percent)
    mask_thre = mask_sorted[pruning_num]
    ### 3. convert subgraph to whole graph
    select_index = torch.abs(mask) < mask_thre
    select_real_index = edges_orig_idx[:, select_index.squeeze()]    
    ### 4. pruning whole adj matrix: ogbndataset.adj
    for idx in select_real_index.t():
        ogbndataset.adj[idx[0].item(), idx[1].item()] = 0
    ogbndataset.adj.eliminate_zeros()
    pruning_links = select_real_index.shape[1]
    # print("Subgraph:[{}] Pruning:[{}] links, Remain:[{}/{}={:.6f}%]"
    # .format(it + 1,
    #         select_real_index.shape[1],
    #         ogbndataset.adj.nnz, 
    #         model.num_original_edge, 
    #         ogbndataset.adj.nnz * 100 / model.num_original_edge))
    return pruning_links


def save_all(dataset, model, rewind_weight, optimizer, imp_num, epoch, save_path, save_name='default'):
    
    state = {
            'adj': dataset.adj,
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
    return state


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

#### 123121
def add_mask(model):

    for i in range(28):
        mask_train1 = nn.Parameter(torch.ones_like(model.gcns[i].mlp[0].weight))
        mask_fixed1 = nn.Parameter(torch.ones_like(model.gcns[i].mlp[0].weight), requires_grad=False)
        AddTrainableMask.apply(model.gcns[i].mlp[0], 'weight', mask_train1, mask_fixed1)

        mask_train2 = nn.Parameter(torch.ones_like(model.gcns[i].mlp[3].weight))
        mask_fixed2 = nn.Parameter(torch.ones_like(model.gcns[i].mlp[3].weight), requires_grad=False)
        AddTrainableMask.apply(model.gcns[i].mlp[3], 'weight', mask_train2, mask_fixed2)


def subgradient_update_mask(model, args):

    model.edge_mask.grad.data.add_(args.s1 * torch.sign(model.edge_mask.data))
    for i in range(28):
        model.gcns[i].mlp[0].weight_mask_train.grad.data.add_(args.s2 * torch.sign(model.gcns[i].mlp[0].weight_mask_train.data))
        model.gcns[i].mlp[3].weight_mask_train.grad.data.add_(args.s2 * torch.sign(model.gcns[i].mlp[3].weight_mask_train.data))


def get_soft_mask_distribution(model):

    weight_mask_vector = torch.tensor([]).to(torch.device("cuda:0"))
    for i in range(28):
        weight_mask1 = model.gcns[i].mlp[0].weight_mask_train.flatten()
        nonzero = torch.abs(weight_mask1) > 0
        weight_mask1 = weight_mask1[nonzero]

        weight_mask2 = model.gcns[i].mlp[3].weight_mask_train.flatten()
        nonzero = torch.abs(weight_mask2) > 0
        weight_mask2 = weight_mask2[nonzero]

        weight_mask_vector = torch.cat((weight_mask_vector, weight_mask1))
        weight_mask_vector = torch.cat((weight_mask_vector, weight_mask2))
    
    return weight_mask_vector.detach().cpu()


### 1. set hard mask < percent to zero
def pruning_weight_mask(model, percent):
    ### weight vector
    wei_total = 458752
    wei_mask = get_soft_mask_distribution(model)
    # wei_total = wei_mask.numel() # 458752
    ### sort
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    wei_thre_index = int(wei_total * percent)
    wei_thre = wei_y[wei_thre_index]
    ### 
    total = 0
    remain = 0
    for i in range(28):
        
        model.gcns[i].mlp[0].weight_mask_fixed = nn.Parameter(get_each_mask(model.gcns[i].mlp[0].weight_mask_train, wei_thre), requires_grad=False)
        model.gcns[i].mlp[3].weight_mask_fixed = nn.Parameter(get_each_mask(model.gcns[i].mlp[3].weight_mask_train, wei_thre), requires_grad=False)

        model.gcns[i].mlp[0].weight_mask_train = nn.Parameter(get_each_mask(model.gcns[i].mlp[0].weight_mask_train, wei_thre, ifone=False), requires_grad=True)
        model.gcns[i].mlp[3].weight_mask_train = nn.Parameter(get_each_mask(model.gcns[i].mlp[3].weight_mask_train, wei_thre, ifone=False), requires_grad=True)

        total += model.gcns[i].mlp[0].weight_mask_fixed.numel()
        total += model.gcns[i].mlp[3].weight_mask_fixed.numel()
        remain += model.gcns[i].mlp[0].weight_mask_fixed.detach().sum()
        remain += model.gcns[i].mlp[3].weight_mask_fixed.detach().sum()
    
    return remain, wei_total


def get_each_mask(mask_weight_tensor, threshold, ifone=True):
    if ifone:
        ones  = torch.ones_like(mask_weight_tensor)
        zeros = torch.zeros_like(mask_weight_tensor) 
        mask = torch.where(mask_weight_tensor.abs() > threshold, ones, zeros)
        return mask
    else:
        zeros = torch.zeros_like(mask_weight_tensor) 
        mask = torch.where(mask_weight_tensor.abs() > threshold, mask_weight_tensor, zeros)
        return mask

##### pruning remain mask percent #######
def get_final_mask_epoch(model, rewind_weight, wei_percent):

    wei_mask = get_soft_mask_distribution(model)
    ### sort
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    wei_thre_index = int(wei_total * wei_percent)
    wei_thre = wei_y[wei_thre_index]
    ### create mask 
    
    for i in range(28):
        key_train = 'gcns.{}.mlp.0.weight_mask_train'.format(i)
        key_fixed = 'gcns.{}.mlp.0.weight_mask_fixed'.format(i)
        rewind_weight[key_train] = get_each_mask(model.gcns[i].mlp[0].state_dict()['weight_mask_train'], wei_thre)
        rewind_weight[key_fixed] = rewind_weight[key_train]

    return rewind_weight


def random_pruning(dataset, model, args):
    
    #### random pruning adj
    edge_total = dataset.adj.nnz
    edge_pruned_num = int(edge_total * args.pruning_percent_adj)
    adj_nonzero = dataset.adj.nonzero()
    edge_pruned_index = random.sample([i for i in range(edge_total)], edge_pruned_num)
    
    print("pruning edge ......")
    for index in tqdm(edge_pruned_index):
        x, y = adj_nonzero[0][index], adj_nonzero[1][index]
        dataset.adj[x, y] = 0
    dataset.adj.eliminate_zeros()

    
    #### random pruning weight
    for i in range(args.num_layers):
        
        model.gcns[i].mlp[0].weight_mask_train.requires_grad = False
        model.gcns[i].mlp[3].weight_mask_train.requires_grad = False

        #####  mlp 1 #####
        wei_total = model.gcns[i].mlp[0].weight_mask_train.numel()
        wei_pruned_num = int(wei_total * args.pruning_percent_wei)
        wei_nonzero = model.gcns[i].mlp[0].weight_mask_train.nonzero()
        wei_pruned_index = random.sample([j for j in range(wei_total)], wei_pruned_num)
        wei_pruned_list = wei_nonzero[wei_pruned_index].tolist()

        for ii, (ai, wj) in enumerate(wei_pruned_list):
            model.gcns[i].mlp[0].weight_mask_train[ai][wj] = 0
            model.gcns[i].mlp[0].weight_mask_fixed[ai][wj] = 0

        #####  mlp 3 #####
        wei_total = model.gcns[i].mlp[3].weight_mask_train.numel()
        wei_pruned_num = int(wei_total * args.pruning_percent_wei)
        wei_nonzero = model.gcns[i].mlp[3].weight_mask_train.nonzero()
        wei_pruned_index = random.sample([j for j in range(wei_total)], wei_pruned_num)
        wei_pruned_list = wei_nonzero[wei_pruned_index].tolist()

        for ii, (ai, wj) in enumerate(wei_pruned_list):
            model.gcns[i].mlp[3].weight_mask_train[ai][wj] = 0
            model.gcns[i].mlp[3].weight_mask_fixed[ai][wj] = 0

        model.gcns[i].mlp[0].weight_mask_train.requires_grad = True 
        model.gcns[i].mlp[3].weight_mask_train.requires_grad = True




def print_sparsity(dataset, model):

    adj_spar = dataset.adj.nnz / model.num_original_edge

    weight_total = 0
    weight_nonzero = 0

    for i in range(28):
        weight_total += model.gcns[i].mlp[0].weight_mask_fixed.numel()
        weight_total += model.gcns[i].mlp[3].weight_mask_fixed.numel()
        weight_nonzero += model.gcns[i].mlp[0].weight_mask_fixed.sum().item()
        weight_nonzero += model.gcns[i].mlp[3].weight_mask_fixed.sum().item()
    
    wei_spar = weight_nonzero / weight_total
    print("-" * 100)
    print("Sparsity: Adj:[{:.3f}%] Wei:[{:.3f}%]".format(adj_spar * 100, wei_spar * 100))
    print("-" * 100)
    return adj_spar, wei_spar


def add_trainable_mask_noise(model, c=1e-5):

    for i in range(28):
        add_noise_in_module(model.gcns[i].mlp[0].weight_mask_train, c=c)
        add_noise_in_module(model.gcns[i].mlp[3].weight_mask_train, c=c)
        
def add_noise_in_module(module, c):

    module.requires_grad = False
    rand = (2 * torch.rand(module.shape) - 1) * c
    rand = rand.to(module.device) 
    rand = rand * module
    module.add_(rand)
    module.requires_grad = True
