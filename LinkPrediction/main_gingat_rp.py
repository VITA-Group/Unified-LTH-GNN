"Implementation based on https://github.com/PetarV-/DGI"
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from models import LogReg, GIC_GCN, GIC_GAT, GIC_GIN
from utils import process
import argparse
import pdb
import pruning
import pruning_gin
import pruning_gat
import copy
import dgl
import warnings
warnings.filterwarnings('ignore')

def run_fix_mask(args, imp_num, adj_percent, wei_percent, dataset_dict):

    pruning_gin.setup_seed(args.seed)
    num_clusters = int(args.num_clusters)
    dataset = args.dataset

    batch_size = 1
    patience = 50
    l2_coef = 0.0
    hid_units = 16
    # sparse = True
    sparse = False
    nonlinearity = 'prelu' # special name to separate parameters

    adj = dataset_dict['adj']
    adj_sparse = dataset_dict['adj_sparse']
    features = dataset_dict['features']
    labels = dataset_dict['labels']
    val_edges = dataset_dict['val_edges']
    val_edges_false = dataset_dict['val_edges_false']
    test_edges = dataset_dict['test_edges']
    test_edges_false = dataset_dict['test_edges_false']

    nb_nodes = features.shape[1]
    ft_size = features.shape[2]

    g = dgl.DGLGraph()
    g.add_nodes(nb_nodes)
    adj = adj.tocoo()
    g.add_edges(adj.row, adj.col)

    
    b_xent = nn.BCEWithLogitsLoss()
    b_bce = nn.BCELoss()

    if args.net == 'gin':
        model = GIC_GIN(nb_nodes, ft_size, hid_units, nonlinearity, num_clusters, 100, g)
        pruning_gin.add_mask(model.gcn)
        pruning_gin.random_pruning(model.gcn, adj_percent, wei_percent)
        adj_spar, wei_spar = pruning_gin.print_sparsity(model.gcn)
    elif args.net == 'gat':
        model = GIC_GAT(nb_nodes, ft_size, hid_units, nonlinearity, num_clusters, 100, g)
        g.add_edges(list(range(nb_nodes)), list(range(nb_nodes)))
        pruning_gat.add_mask(model.gcn)
        pruning_gat.random_pruning(model.gcn, adj_percent, wei_percent)
        adj_spar, wei_spar = pruning_gat.print_sparsity(model.gcn)
    else: assert False

    for name, param in model.named_parameters():
        if 'mask' in name:
            param.requires_grad = False
            #print("NAME:{}\tSHAPE:{}\tGRAD:{}".format(name, param.shape, param.requires_grad))

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=l2_coef)
    model.cuda()
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc':0}
    for epoch in range(1, args.fix_epoch + 1):
        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]
        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()

        logits, logits2 = model(features, shuf_fts, 
                                          g, 
                                          sparse, None, None, None, 100) 
        loss = 0.5 * b_xent(logits, lbl) + 0.5 * b_xent(logits2, lbl) 
        loss.backward()
        optimiser.step()

        with torch.no_grad():
            acc_val, _ = pruning.test(model, features, 
                                             g, 
                                             sparse, 
                                             adj_sparse, 
                                             val_edges, 
                                             val_edges_false)
            acc_test, _ = pruning.test(model, features, 
                                              g, 
                                              sparse, 
                                              adj_sparse, 
                                              test_edges, 
                                              test_edges_false)
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                
        print("RP [{}] ({} {} FIX Mask) Epoch:[{}/{}], Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}] | Adj:[{:.2f}%] Wei:[{:.2f}%]"
             .format(imp_num,
                            args.net,
                            args.dataset,
                            epoch, 
                            args.fix_epoch, 
                            loss, 
                            acc_val * 100, 
                            acc_test * 100, 
                            best_val_acc['val_acc'] * 100,
                            best_val_acc['test_acc'] * 100,
                            best_val_acc['epoch'],
                            adj_spar,
                            wei_spar))

    print("syd final: RP[{}] ({} {} FIX Mask) | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}] | Adj:[{:.2f}%] Wei:[{:.2f}%]"
             .format(imp_num,
                            args.net,
                            args.dataset, 
                            best_val_acc['val_acc'] * 100,
                            best_val_acc['test_acc'] * 100,
                            best_val_acc['epoch'],
                            adj_spar,
                            wei_spar))

def parser_loader():

    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--s1', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--s2', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--mask_epoch', type=int, default=300)
    parser.add_argument('--fix_epoch', type=int, default=300)
    parser.add_argument('--pruning_percent_wei', type=float, default=0.2)
    parser.add_argument('--pruning_percent_adj', type=float, default=0.05)
    parser.add_argument('--net',  type=str, default='',help='')

    parser.add_argument('--epochs', type=int, default=2000, help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--seed', type=int, default=1234, help='')
    parser.add_argument('--dataset',  type=str, default='cora',help='')
    parser.add_argument('--b', dest='beta', type=int, default=100,help='')
    parser.add_argument('--c', dest='num_clusters', type=float, default=128,help='')
    parser.add_argument('--a', dest='alpha', type=float, default=0.5,help='')
    parser.add_argument('--test_rate', dest='test_rate', type=float, default=0.1,help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parser_loader()
    pruning_gin.print_args(args)

    dataset = args.dataset
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    adj_sparse = adj
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = process.mask_test_edges(adj, test_frac=args.test_rate, val_frac=0.05)
    adj = adj_train
    features, _ = process.preprocess_features(features)
    features = torch.FloatTensor(features[np.newaxis]).cuda()
    # adj = torch.FloatTensor(adj.todense()).cuda()
    labels = torch.FloatTensor(labels[np.newaxis]).cuda()

    dataset_dict = {}
    dataset_dict['adj'] = adj
    dataset_dict['adj_sparse'] = adj_sparse
    dataset_dict['features'] = features
    dataset_dict['labels'] = labels
    dataset_dict['val_edges'] = val_edges
    dataset_dict['val_edges_false'] = val_edges_false
    dataset_dict['test_edges'] = test_edges
    dataset_dict['test_edges_false'] = test_edges_false

    percent_list = [(1 - (1 - args.pruning_percent_adj) ** (i + 1), 1 - (1 - args.pruning_percent_wei) ** (i + 1)) for i in range(20)]

    for imp_num, (adj_percent, wei_percent) in enumerate(percent_list):
        run_fix_mask(args, imp_num + 1, adj_percent, wei_percent, dataset_dict)
        