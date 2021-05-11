import os
import random
import argparse

import torch
import torch.nn as nn
import numpy as np

import net as net
from utils import load_data
from sklearn.metrics import f1_score
import pdb
import pruning
import copy
import utils
import warnings
warnings.filterwarnings('ignore')

def run_fix_mask(args, index, rewind_weight_mask, seed):

    adj = np.load("./ADMM/admm_{}/adj_{}.npy".format(args['dataset'], index))
    adj = utils.normalize_adj(adj)
    adj = utils.sparse_mx_to_torch_sparse_tensor(adj)

    pruning.setup_seed(seed)
    _, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = adj.to_dense()

    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()

    net_gcn = net.net_gcn_baseline(embedding_dim=args['embedding_dim'])
    pruning.add_mask(net_gcn)
    net_gcn = net_gcn.cuda()
    net_gcn.load_state_dict(rewind_weight_mask)
    wei_spar = pruning.print_weight_sparsity(net_gcn)
    
    for name, param in net_gcn.named_parameters():
        if 'mask' in name:
            param.requires_grad = False
            print("NAME:{}\tSHAPE:{}\tGRAD:{}".format(name, param.shape, param.requires_grad))

    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc': 0, 'wei_spar' : 0}
    best_val_acc['wei_spar'] = wei_spar
    for epoch in range(200):

        optimizer.zero_grad()
        output = net_gcn(features, adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
            acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch
 
        print("(ADMM Fix Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                 .format(epoch, acc_val * 100, acc_test * 100, 
                                best_val_acc['val_acc'] * 100, 
                                best_val_acc['test_acc'] * 100, 
                                best_val_acc['epoch']))

    return best_val_acc


def run_get_admm_weight_mask(args, index, wei_percent, seed):

    adj = np.load("./ADMM/admm_{}/adj_{}.npy".format(args['dataset'], index))
    adj = utils.normalize_adj(adj)
    adj = utils.sparse_mx_to_torch_sparse_tensor(adj)

    pruning.setup_seed(seed)
    _, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = adj.to_dense()

    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()
    
    net_gcn = net.net_gcn_baseline(embedding_dim=args['embedding_dim'])
    pruning.add_mask(net_gcn)
    net_gcn = net_gcn.cuda()

    for name, param in net_gcn.named_parameters():
        if 'mask' in name:
            param.requires_grad = False
            print("NAME:{}\tSHAPE:{}\tGRAD:{}".format(name, param.shape, param.requires_grad))
    
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc':0}
    rewind_weight = copy.deepcopy(net_gcn.state_dict())

    for epoch in range(args['total_epoch']):
        
        optimizer.zero_grad()
        output = net_gcn(features, adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()

        optimizer.step()
        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
            acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                best_epoch_mask = pruning.get_final_weight_mask_epoch(net_gcn, wei_percent=wei_percent)

            print("(ADMM Get Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                 .format(epoch, acc_val * 100, acc_test * 100, 
                                best_val_acc['val_acc'] * 100,  
                                best_val_acc['test_acc'] * 100,
                                best_val_acc['epoch']))

    return best_epoch_mask, rewind_weight


def parser_loader():
    parser = argparse.ArgumentParser(description='Self-Supervised GCN')
    ###### Unify pruning settings #######
    parser.add_argument('--total_epoch', type=int, default=300)
    parser.add_argument('--pruning_percent_wei', type=float, default=0.2)
    parser.add_argument('--pruning_percent_adj', type=float, default=0.05)
    ###### Others settings #######
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703,16,6])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    return parser


if __name__ == "__main__":

    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)
    ####################81.9 ##############72.1######################
    # seed_dict = {'cora': 3946, 'citeseer': 2239} # DIM: 16
    seed_dict = {'cora': 2377, 'citeseer': 4428, 'pubmed': 3333} # DIM: 512, cora: 2829: 81.9, 2377: 81.1    | cite 4417: 72.1,  4428: 71.3
    seed = seed_dict[args['dataset']]
    
    percent_list = [(1 - (1 - args['pruning_percent_adj']) ** (i + 1), 1 - (1 - args['pruning_percent_wei']) ** (i + 1)) for i in range(20)]
    
    for index, (_, wei_percent) in enumerate(percent_list):
        
        final_mask_dict, rewind_weight = run_get_admm_weight_mask(args, index + 1, wei_percent, seed)

        rewind_weight['net_layer.0.weight_mask_train'] = final_mask_dict['weight1_mask']
        rewind_weight['net_layer.0.weight_mask_fixed'] = final_mask_dict['weight1_mask']
        rewind_weight['net_layer.1.weight_mask_train'] = final_mask_dict['weight2_mask']
        rewind_weight['net_layer.1.weight_mask_fixed'] = final_mask_dict['weight2_mask']
        
        best_val_acc = run_fix_mask(args, index + 1, rewind_weight, seed)
        print("=" * 120)
        print("syd : INDEX:[{}], Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] Wei:[{:.2f}%]"
            .format(index + 1, best_val_acc['val_acc'] * 100, best_val_acc['epoch'], best_val_acc['test_acc'] * 100, best_val_acc['wei_spar']))
        print("=" * 120)
