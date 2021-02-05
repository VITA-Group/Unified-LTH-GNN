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
from gnns_clean.gin_net import GINNet
from gnns_clean.gat_net import GATNet
import pruning


def run(args):

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
        net_gcn = GINNet(args['embedding_dim'])
    else:
        net_gcn = GATNet(args['embedding_dim'])
        g.add_edges(list(range(node_num)), list(range(node_num)))
    net_gcn = net_gcn.cuda()
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc': 0}
    for epoch in range(args['total_epoch']):

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

        print("(Baseline) Epoch:[{}] LOSS:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                 .format(epoch, loss,
                                acc_val * 100, 
                                acc_test * 100, 
                                best_val_acc['val_acc'] * 100, 
                                best_val_acc['test_acc'] * 100, 
                                best_val_acc['epoch']))

    print("syd final: [{},{}] (Baseline) Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                 .format(   args['dataset'],
                            args['net'],
                            best_val_acc['val_acc'] * 100, 
                            best_val_acc['test_acc'] * 100, 
                            best_val_acc['epoch']))

def parser_loader():
    parser = argparse.ArgumentParser(description='Self-Supervised GCN')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703,16,6])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--net', type=str, default='')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--total_epoch', type=int, default=1000)

    return parser


if __name__ == "__main__":

    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)
    for i in range(50):
        args['seed'] = 50 + i
        run(args)
    