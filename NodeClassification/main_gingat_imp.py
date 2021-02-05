import os
import random
import argparse

import torch
import torch.nn as nn
import numpy as np

import net as net
from utils import load_data, load_adj_raw
from sklearn.metrics import f1_score

import dgl
from gnns.gin_net import GINNet
from gnns.gat_net import GATNet
import pruning
import pruning_gin
import pruning_gat
import pdb
import warnings
warnings.filterwarnings('ignore')
import copy

def run_fix_mask(args, imp_num, rewind_weight_mask):

    pruning.setup_seed(args['seed'])
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = load_adj_raw(args['dataset'])

    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    g = dgl.DGLGraph()
    g.add_nodes(node_num)
    adj = adj.tocoo()
    g.add_edges(adj.row, adj.col)
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()

    if args['net'] == 'gin':
        net_gcn = GINNet(args['embedding_dim'], g)
        pruning_gin.add_mask(net_gcn)
    elif args['net'] == 'gat':
        net_gcn = GATNet(args['embedding_dim'], g)
        g.add_edges(list(range(node_num)), list(range(node_num)))
        pruning_gat.add_mask(net_gcn)
    else: assert False

    net_gcn = net_gcn.cuda()
    net_gcn.load_state_dict(rewind_weight_mask)

    if args['net'] == 'gin':
        adj_spar, wei_spar = pruning_gin.print_sparsity(net_gcn)
    else:
        adj_spar, wei_spar = pruning_gat.print_sparsity(net_gcn)

    for name, param in net_gcn.named_parameters():
        if 'mask' in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc': 0}

    for epoch in range(args['fix_epoch']):

        optimizer.zero_grad()
        output = net_gcn(g, features, 0, 0)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            net_gcn.eval()
            output = net_gcn(g, features, 0, 0)
            acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch

        print("IMP[{}] (Fix Mask) Epoch:[{}] LOSS:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
               .format(imp_num, epoch, 
                                args['fix_epoch'],
                                loss,
                                acc_val * 100, 
                                acc_test * 100, 
                                best_val_acc['val_acc'] * 100, 
                                best_val_acc['test_acc'] * 100, 
                                best_val_acc['epoch']))

    print("syd final: [{},{}] IMP[{}] (Fix Mask) Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}] | Adj:[{:.2f}%] Wei:[{:.2f}%]"
                 .format(   args['dataset'],
                            args['net'],
                            imp_num,
                            best_val_acc['val_acc'] * 100, 
                            best_val_acc['test_acc'] * 100, 
                            best_val_acc['epoch'],
                            adj_spar,
                            wei_spar))


def run_get_mask(args, imp_num, rewind_weight_mask=None):

    pruning.setup_seed(args['seed'])
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = load_adj_raw(args['dataset'])
    
    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    g = dgl.DGLGraph()
    g.add_nodes(node_num)
    adj = adj.tocoo()
    
    g.add_edges(adj.row, adj.col)
    features = features.cuda()
    labels = labels.cuda()

    loss_func = nn.CrossEntropyLoss()

    if args['net'] == 'gin':
        net_gcn = GINNet(args['embedding_dim'], g)
        pruning_gin.add_mask(net_gcn)
    elif args['net'] == 'gat':
        net_gcn = GATNet(args['embedding_dim'], g)
        g.add_edges(list(range(node_num)), list(range(node_num)))
        pruning_gat.add_mask(net_gcn)
    else: assert False

    net_gcn = net_gcn.cuda()

    if rewind_weight_mask:
        net_gcn.load_state_dict(rewind_weight_mask)
    
    if args['net'] == 'gin':
        pruning_gin.add_trainable_mask_noise(net_gcn, c=1e-5)
        adj_spar, wei_spar = pruning_gin.print_sparsity(net_gcn)
    else:
        pruning_gat.add_trainable_mask_noise(net_gcn, c=1e-5)
        adj_spar, wei_spar = pruning_gat.print_sparsity(net_gcn)

    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc': 0}

    rewind_weight = copy.deepcopy(net_gcn.state_dict())

    for epoch in range(args['mask_epoch']):

        optimizer.zero_grad()
        output = net_gcn(g, features, 0, 0)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        if args['net'] == 'gin':
            pruning_gin.subgradient_update_mask(net_gcn, args) # l1 norm
        else:
            pruning_gat.subgradient_update_mask(net_gcn, args) # l1 norm
            
        optimizer.step()
        with torch.no_grad():
            net_gcn.eval()
            output = net_gcn(g, features, 0, 0)
            acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch

                if args['net'] == 'gin':
                    rewind_weight, adj_spar, wei_spar = pruning_gin.get_final_mask_epoch(net_gcn, rewind_weight, args) 
                else:
                    rewind_weight, adj_spar, wei_spar = pruning_gat.get_final_mask_epoch(net_gcn, rewind_weight, args)
                 
        print("IMP[{}] (Get Mask) Epoch:[{}/{}] LOSS:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}] | Adj:[{:.2f}%] Wei:[{:.2f}%]"
               .format(imp_num, epoch, 
                                args['mask_epoch'],
                                loss,
                                acc_val * 100, 
                                acc_test * 100, 
                                best_val_acc['val_acc'] * 100, 
                                best_val_acc['test_acc'] * 100, 
                                best_val_acc['epoch'],
                                adj_spar,
                                wei_spar))

    return rewind_weight


def parser_loader():
    parser = argparse.ArgumentParser(description='Self-Supervised GCN')
    ###### Unify pruning settings #######
    parser.add_argument('--s1', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--s2', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--mask_epoch', type=int, default=300)
    parser.add_argument('--fix_epoch', type=int, default=300)
    parser.add_argument('--pruning_percent_wei', type=float, default=0.1)
    parser.add_argument('--pruning_percent_adj', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703,16,6])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--net', type=str, default='')
    parser.add_argument('--seed', type=int, default=666)
    return parser


if __name__ == "__main__":

    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)
    
    rewind_weight = None
    for imp in range(1, 21):
        
        rewind_weight = run_get_mask(args, imp, rewind_weight)
        run_fix_mask(args, imp, rewind_weight)
        
    