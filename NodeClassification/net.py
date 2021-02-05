import torch
import torch.nn as nn
import pdb
import copy
import utils

class net_gcn(nn.Module):

    def __init__(self, embedding_dim, adj):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.adj_nonzero = torch.nonzero(adj, as_tuple=False).shape[0]
        self.adj_mask1_train = nn.Parameter(self.generate_adj_mask(adj))
        self.adj_mask2_fixed = nn.Parameter(self.generate_adj_mask(adj), requires_grad=False)
        self.normalize = utils.torch_normalize_adj
    
    def forward(self, x, adj, val_test=False):
        
        adj = torch.mul(adj, self.adj_mask1_train)
        adj = torch.mul(adj, self.adj_mask2_fixed)
        adj = self.normalize(adj)
        #adj = torch.mul(adj, self.adj_mask2_fixed)
        for ln in range(self.layer_num):
            x = torch.mm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x

    def generate_adj_mask(self, input_adj):
        
        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)
        return mask


class net_gcn_admm(nn.Module):

    def __init__(self, embedding_dim, adj):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.adj_nonzero = torch.nonzero(adj, as_tuple=False).shape[0]
        self.adj_layer1 = nn.Parameter(copy.deepcopy(adj), requires_grad=True)
        self.adj_layer2 = nn.Parameter(copy.deepcopy(adj), requires_grad=True)
        
    def forward(self, x, adj, val_test=False):

        for ln in range(self.layer_num):
            if ln == 0:
                x = torch.mm(self.adj_layer1, x)
            elif ln == 1:
                x = torch.mm(self.adj_layer2, x)
            else:
                assert False
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x

    # def forward(self, x, adj, val_test=False):

    #     for ln in range(self.layer_num):
    #         x = torch.mm(self.adj_list[ln], x)
    #         x = self.net_layer[ln](x)
    #         if ln == self.layer_num - 1:
    #             break
    #         x = self.relu(x)
    #         if val_test:
    #             continue
    #         x = self.dropout(x)
    #     return x

    def generate_adj_mask(self, input_adj):
        
        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)
        return mask

class net_gcn_baseline(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, adj, val_test=False):

        for ln in range(self.layer_num):
            x = torch.mm(adj, x)
            # x = torch.spmm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x


class net_gcn_multitask(nn.Module):

    def __init__(self, embedding_dim, ss_dim):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.ss_classifier = nn.Linear(embedding_dim[-2], ss_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        

    def forward(self, x, adj, val_test=False):

        x_ss = x

        for ln in range(self.layer_num):
            x = torch.spmm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)

        if not val_test:
            for ln in range(self.layer_num):
                x_ss = torch.spmm(adj, x_ss)
                if ln == self.layer_num - 1:
                    break
                x_ss = self.net_layer[ln](x_ss)
                x_ss = self.relu(x_ss)
                x_ss = self.dropout(x_ss)
            x_ss = self.ss_classifier(x_ss)

        return x, x_ss

