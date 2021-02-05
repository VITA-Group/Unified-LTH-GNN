import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
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

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        
        ### edges.src['z']: edge src node feature
        ### edges.data['e']: edge feature
        return {'z': edges.src['z'], 'e': edges.data['e']} # this dict all save in message box

    def reduce_func(self, nodes):
       
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, snorm_n, train_mask, fixed_mask):

        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention) # The function to generate new edge features
        ### pruning edges 
        g.edata['e'] = g.edata['e'] * train_mask * fixed_mask
        g.update_all(self.message_func, self.reduce_func)

        h = g.ndata['h']
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)

        if not self.heads == 1:
            h = F.elu(h)
            h = F.dropout(h, self.dropout, training=self.training)
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
        for i in range(num_heads): # 8
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
