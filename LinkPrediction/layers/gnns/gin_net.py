import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import pdb
"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

# from gnns_clean.gin_layer import GINLayer, ApplyNodeFunc, MLP

class GINNet(nn.Module):
    
    def __init__(self, net_params, graph):
        super().__init__()
        in_dim = net_params[0]
        hidden_dim = net_params[1]
        n_classes = net_params[2]
        dropout = 0.5
        self.n_layers = 2
        self.edge_num = graph.all_edges()[0].numel()
        n_mlp_layers = 1               # GIN
        learn_eps = True              # GIN
        neighbor_aggr_type = 'mean' # GIN
        graph_norm = False      
        batch_norm = False
        residual = False
        self.n_classes = n_classes
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        for layer in range(self.n_layers):
            if layer == 0:
                mlp = MLP(n_mlp_layers, in_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, n_classes)
                
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, graph_norm, batch_norm, residual, 0, learn_eps))
        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = nn.Linear(hidden_dim, n_classes, bias=False)
        for m in self.modules():
            self.weights_init(m)

        self.adj_mask1_train = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=True)
        self.adj_mask2_fixed = nn.Parameter(torch.ones(self.edge_num, 1), requires_grad=False)
        
    def forward(self, g, h, snorm_n, snorm_e):
        # list of hidden representation at each layer (including input)
        g.edata['mask'] = self.adj_mask1_train * self.adj_mask2_fixed
        hidden_rep = []
        h = h.squeeze()
        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n)
            hidden_rep.append(h)

        # score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2
        score_over_layer = (self.linears_prediction(hidden_rep[0]) + hidden_rep[1]) / 2
        score_over_layer = score_over_layer.unsqueeze(0)

        return score_over_layer
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    


"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""
msg_mask = fn.src_mul_edge('h', 'mask', 'm')
# msg_mask = fn.u_mul_e('h', 'mask', 'm')
msg_orig = fn.copy_u('h', 'm')
class GINLayer(nn.Module):
    """
    [!] code adapted from dgl implementation of GINConv

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggr_type :
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    out_dim :
        Rquired for batch norm layer; should match out_dim of apply_func if not None.
    dropout :
        Required for dropout of output features.
    graph_norm : 
        boolean flag for output features normalization w.r.t. graph sizes.
    batch_norm :
        boolean flag for batch_norm layer.
    residual :
        boolean flag for using residual connection.
    init_eps : optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter.
    
    """
    def __init__(self, apply_func, aggr_type, dropout, graph_norm, batch_norm, residual=False, init_eps=0, learn_eps=False):
        super().__init__()
        self.apply_func = apply_func
        
        if aggr_type == 'sum':
            self._reducer = fn.sum
        elif aggr_type == 'max':
            self._reducer = fn.max
        elif aggr_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggr_type))
            
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout
        self.act = nn.PReLU()
        in_dim = apply_func.mlp.input_dim
        out_dim = apply_func.mlp.output_dim
        
        if in_dim != out_dim:
            self.residual = False
            
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))
            
        self.bn_node_h = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, snorm_n):

        g.ndata['h'] = h
        g.update_all(msg_mask, self._reducer('m', 'neigh'))
        # g.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'neigh'))
        h = (1 + self.eps) * h + g.ndata['neigh']
        if self.apply_func is not None:
            h = self.apply_func(h)
        h = self.act(h)
        return h
    
    
class ApplyNodeFunc(nn.Module):
    """
        This class is used in class GINNet
        Update the node feature hv with MLP
    """
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, h):
        h = self.mlp(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super().__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.linears[i](h))
            return self.linears[-1](h)
