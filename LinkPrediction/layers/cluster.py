'''
    pytorch (differentiable) implementation of soft k-means clustering. 
    Modified from https://github.com/bwilder0/clusternet
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import sklearn.cluster
#from functional import  pairwise_distances

def cluster(data, k, temp, num_iter, init, cluster_temp):
    
    eps = 1e-4
    cuda0 = torch.cuda.is_available()#False
    if cuda0:
        mu = init.cuda()
        data = data.cuda()
        cluster_temp = cluster_temp.cuda()
    else:
        mu = init
    n = data.shape[0]
    d = data.shape[1]
#    
    data = data / (data.norm(dim=1)[:, None] + eps)
    # data = torch.where(torch.isnan(data), torch.full_like(data, 0.1), data)

    for t in range(num_iter):
        
        mu = mu / (mu.norm(dim=1)[:, None] + eps)
        dist = torch.mm(data, mu.transpose(0,1))
        #cluster responsibilities via softmax
        r = F.softmax(cluster_temp * dist, dim=1)
        #total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        #mean of points in each cluster weighted by responsibility
        cluster_mean = r.t() @ data
        #update cluster means
        new_mu = torch.diag(1 / (cluster_r + eps)) @ cluster_mean
        mu = new_mu

    r = F.softmax(cluster_temp * dist, dim=1)
    return mu, r

class Clusterator(nn.Module):
    '''
    The ClusterNet architecture. The first step is a 2-layer GCN to generate embeddings.
    The output is the cluster means mu and soft assignments r, along with the 
    embeddings and the the node similarities (just output for debugging purposes).
    
    The forward pass inputs are x, a feature matrix for the nodes, and adj, a sparse
    adjacency matrix. The optional parameter num_iter determines how many steps to 
    run the k-means updates for.
    '''
    def __init__(self, nout, K):
        super(Clusterator, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.nout = nout
        self.init =  torch.rand(self.K, nout)
        
    def forward(self, embeds, cluster_temp, num_iter=10):
        
        mu_init, _ = cluster(embeds, self.K, 1, num_iter, cluster_temp = torch.tensor(cluster_temp), init = self.init)
        #self.init = mu_init.clone().detach()
        mu, r = cluster(embeds, self.K, 1, 1, cluster_temp = torch.tensor(cluster_temp), init = mu_init.clone().detach())
        
        return mu, r