import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import pdb

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
# from gnns_clean.gat_layer import GATLayer
# from gnns_clean.mlp_readout_layer import MLPReadout

class GATNet(nn.Module):

    def __init__(self, net_params, graph):
        super().__init__()

        in_dim_node = net_params[0] # node_dim (feat is an integer)
        hidden_dim = net_params[1]
        out_dim = net_params[2]
        n_classes = net_params[2]
        num_heads = 1
        dropout = 0.6
        n_layers = 1
        self.edge_num = graph.number_of_edges() + graph.number_of_nodes()
        self.graph_norm = False
        self.batch_norm = False
        self.residual = False
        self.dropout = dropout
        self.n_classes = n_classes
        
        self.layers = nn.ModuleList([GATLayer(in_dim_node, hidden_dim, num_heads,
                                              dropout, self.graph_norm, 
                                                       self.batch_norm, 
                                                       self.residual) for _ in range(n_layers)])
        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, 0, self.graph_norm, self.batch_norm, self.residual))

        for m in self.modules():
            self.weights_init(m)

        self.adj_mask1_train = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=True)
        self.adj_mask2_fixed = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=False)

    def forward(self, g, h, snorm_n, snorm_e):

        # GAT
        g = g.local_var()
        h = h.squeeze()
        for conv in self.layers:
            h = conv(g, h, snorm_n, self.adj_mask1_train, self.adj_mask2_fixed)
        h = h.unsqueeze(0)
        return h
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""

class GATHeadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, heads):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.heads = heads
        self.act = nn.PReLU()
    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        #alpha = F.dropout(alpha, self.dropout, training=self.training)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, snorm_n, train_mask, fixed_mask):
        
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.edata['e'] = g.edata['e'] * train_mask * fixed_mask
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        if not self.heads == 1:
            h = self.act(h)
            # h = F.elu(h)
            #h = F.dropout(h, self.dropout, training=self.training)
        return h

class GATLayer(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, graph_norm, batch_norm, residual=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual
        
        if in_dim != (out_dim*num_heads):
            self.residual = False
        
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATHeadLayer(in_dim, out_dim, dropout, graph_norm, batch_norm, num_heads))
        self.merge = 'cat' 

    def forward(self, g, h, snorm_n, train_mask, fixed_mask):
        h_in = h # for residual connection
        head_outs = [attn_head(g, h, snorm_n, train_mask, fixed_mask) for attn_head in self.heads]
        
        if self.merge == 'cat':
            h = torch.cat(head_outs, dim=1)
        else:
            h = torch.mean(torch.stack(head_outs))
        
        if self.residual:
            h = h_in + h # residual connection
        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
