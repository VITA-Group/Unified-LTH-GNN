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


def run(args, seed):

    setup_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = load_adj_raw(args['dataset'])
    
    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    g = dgl.DGLGraph()
    g.add_nodes(node_num)
    adj = adj.tocoo()
    g.add_edges(adj.row, adj.col)
    features = features.cuda()
    
    '''
    adj = adj.cuda()
    features = features.cuda()
    '''
    labels = labels.cuda()

    loss_func = nn.CrossEntropyLoss()
    early_stopping = 10

    if args['net'] == 'gin':
        net_gcn = GINNet(args['embedding_dim'])
    else:
        net_gcn = GATNet(args['embedding_dim'])
        g.add_edges(list(range(node_num)), list(range(node_num)))
    net_gcn = net_gcn.cuda()
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    loss_val = []
    for epoch in range(1000):

        optimizer.zero_grad()
        output = net_gcn(g, features, 0, 0)
        loss = loss_func(output[idx_train], labels[idx_train])
        # print('epoch', epoch, 'loss', loss.data)
        loss.backward()
        optimizer.step()

        # validation
        with torch.no_grad():
            net_gcn.eval()
            output = net_gcn(g, features, 0, 0)
            loss_val.append(loss_func(output[idx_val], labels[idx_val]).cpu().numpy())
            # print('val acc', f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro'))

        # early stopping
        if epoch > early_stopping and loss_val[-1] > np.mean(loss_val[-(early_stopping+1):-1]):
            break

    # test
    with torch.no_grad():
        net_gcn.eval()
        output = net_gcn(g, features, 0, 0)
        acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
        acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')

    return acc_val, acc_test


def parser_loader():
    parser = argparse.ArgumentParser(description='Self-Supervised GCN')
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703,16,6])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--net', type=str, default='gin')
    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)

    acc_val = np.zeros(50)
    acc_test = np.zeros(50)
    for seed in range(50):
        acc_val[seed], acc_test[seed] = run(args, seed)
        print('seed', seed, 'val', acc_val[seed], 'test', acc_test[seed])

    print('finish')
    print('val mean', acc_val.mean(), 'val std', acc_val.std())
    print('test mean', acc_test.mean(), 'test std', acc_test.std())