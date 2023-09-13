import __init__
import torch
import torch.nn as nn
from gcn_lib.sparse.torch_vertex import GENConv
from gcn_lib.sparse.torch_nn import norm_layer
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging


class DeeperGCN(torch.nn.Module):
    def __init__(self, args):
        super(DeeperGCN, self).__init__()

        self.edge_num = 2358104
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.block = args.block

        self.checkpoint_grad = False

        hidden_channels = args.hidden_channels
        conv = args.conv
        aggr = args.gcn_aggr

        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale

        norm = args.norm
        mlp_layers = args.mlp_layers

        if self.num_layers > 7:
            self.checkpoint_grad = True

        print('The number of layers {}'.format(self.num_layers),
              'Aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))

        # if self.block == 'res+':
        #     print('LN/BN->ReLU->GraphConv->Res')
        # elif self.block == 'res':
        #     print('GraphConv->LN/BN->ReLU->Res')
        # elif self.block == 'dense':
        #     raise NotImplementedError('To be implemented')
        # elif self.block == "plain":
        #     print('GraphConv->LN/BN->ReLU')
        # else:
        #     raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.edge_mask1_train = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=True)
        self.edge_mask2_fixed = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=False)

        for layer in range(self.num_layers):

            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')

            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

    def forward(self,  x, edge_index):

        h = x
        if self.block == 'res+':
            
            h = self.gcns[0](h, self.edge_mask1_train, self.edge_mask2_fixed, edge_index)
            # h = self.gcns[0](h, edge_index)

            for layer in range(1, self.num_layers):
                h1 = self.norms[layer - 1](h)
                h2 = F.relu(h1)
                h2 = F.dropout(h2, p=self.dropout, training=self.training)

                if self.checkpoint_grad:
                    # res = checkpoint(self.gcns[layer], h2, edge_index)
                    res = checkpoint(self.gcns[layer], h2, self.edge_mask1_train, self.edge_mask2_fixed, edge_index)
                    h = res + h
                else:
                    h = self.gcns[layer](h2, self.edge_mask1_train, self.edge_mask2_fixed, edge_index) + h
                    # h = self.gcns[layer](h2, edge_index) + h
            # may remove relu(), the learnt embeddings should not be restricted by positive value
            h = F.relu(self.norms[self.num_layers - 1](h))
            h = F.dropout(h, p=self.dropout, training=self.training)

        else:
            raise Exception('Unknown block Type')

        return h



class LinkPredictor(torch.nn.Module):
    def __init__(self, args):
        super(LinkPredictor, self).__init__()

        in_channels = args.hidden_channels
        hidden_channels = args.hidden_channels
        out_channels = args.num_tasks
        num_layers = args.lp_num_layers
        norm = args.lp_norm

        if norm.lower() == 'none':
            self.norms = None
        else:
            self.norms = torch.nn.ModuleList()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.norms is not None:
                self.norms.append(norm_layer(norm, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = args.dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            if self.norms is not None:
                x = self.norms(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

