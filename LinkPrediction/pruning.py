import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score



def test(model, features, adj, sparse, adj_sparse, test_edges, test_edges_false):

    eps = 1e-4
    embeds, _,_, S= model.embed(features, adj, sparse, None, 100)
    embs = embeds[0, :]
    embs = embs / (embs.norm(dim=1)[:, None] + eps)
    sc_roc, sc_ap = get_roc_score(test_edges, test_edges_false, embs.cpu().detach().numpy(), adj_sparse)
    return sc_roc, sc_ap

def get_roc_score(edges_pos, edges_neg, embeddings, adj_sparse):
    "from https://github.com/tkipf/gae"
    
    score_matrix = np.dot(embeddings, embeddings.T)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]])) # predicted score
        pos.append(adj_sparse[edge[0], edge[1]]) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]])) # predicted score
        neg.append(adj_sparse[edge[0], edge[1]]) # actual value (0 for negative)
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    
    #print(preds_all, labels_all )
    
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score


def torch_normalize_adj(adj):
    adj = adj + torch.eye(adj.shape[0]).cuda()
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt).cuda()
    return adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)