"Implementation based on https://github.com/PetarV-/DGI"
import torch
import torch.nn as nn
from layers import GCN, GINNet, GATNet, AvgReadout, Discriminator, Discriminator_cluster, Clusterator
from layers import net_gcn_baseline
import torch.nn.functional as F
import numpy as np
import pdb

class GIC_GCN(nn.Module):
    def __init__(self,n_nb, n_in, n_h, activation, num_clusters, beta, adj):
        super(GIC_GCN, self).__init__()

        self.gcn = net_gcn_baseline(embedding_dim=[n_in, 512, n_h], adj=adj)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.disc_c = Discriminator_cluster(n_h,n_h,n_nb,num_clusters)
        self.beta = beta
        self.cluster = Clusterator(n_h,num_clusters)
        
    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2, cluster_temp):
        
        h_1 = self.gcn(seq1, adj)
        h_2 = self.gcn(seq2, adj)
        self.beta = cluster_temp

        Z, S = self.cluster(h_1[-1,:,:], cluster_temp)
        Z_t = S @ Z
        c2 = Z_t
        
        c2 = self.sigm(c2)
        
        c = self.read(h_1, msk)
        c = self.sigm(c) 
        c_x = c.unsqueeze(1)
        c_x = c_x.expand_as(h_1)
        
        ret = self.disc(c_x, h_1, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc_c(c2, c2,h_1[-1,:,:], h_1[-1,:,:] ,h_2[-1,:,:], S , samp_bias1, samp_bias2)
        return ret, ret2 

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk, cluster_temp):

        h_1 = self.gcn(seq, adj)
        c = self.read(h_1, msk)
        Z, S = self.cluster(h_1[-1,:,:], self.beta)
        H = S@Z
        return h_1.detach(), H.detach(), c.detach(), Z.detach()



class GIC_GIN(nn.Module):
    def __init__(self,n_nb, n_in, n_h, activation, num_clusters, beta, graph):
        super(GIC_GIN, self).__init__()

        self.gcn = GINNet(net_params=[n_in, 512, n_h], graph=graph)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.disc_c = Discriminator_cluster(n_h,n_h,n_nb,num_clusters)
        self.beta = beta
        self.cluster = Clusterator(n_h,num_clusters)
        
    def forward(self, seq1, seq2, g, sparse, msk, samp_bias1, samp_bias2, cluster_temp):
        
        h_1 = self.gcn(g, seq1, 0, 0)
        h_2 = self.gcn(g, seq2, 0, 0)
        self.beta = cluster_temp

        Z, S = self.cluster(h_1[-1,:,:], cluster_temp)
        Z_t = S @ Z
        c2 = Z_t
        
        c2 = self.sigm(c2)
        
        c = self.read(h_1, msk)
        c = self.sigm(c) 
        c_x = c.unsqueeze(1)
        c_x = c_x.expand_as(h_1)
        
        ret = self.disc(c_x, h_1, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc_c(c2, c2,h_1[-1,:,:], h_1[-1,:,:] ,h_2[-1,:,:], S , samp_bias1, samp_bias2)
        return ret, ret2 

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk, cluster_temp):
        
        h_1 = self.gcn(adj, seq, 0, 0)
        c = self.read(h_1, msk)
        Z, S = self.cluster(h_1[-1,:,:], self.beta)
        H = S@Z
        return h_1.detach(), H.detach(), c.detach(), Z.detach()


class GIC_GAT(nn.Module):
    def __init__(self,n_nb, n_in, n_h, activation, num_clusters, beta, graph):
        super(GIC_GAT, self).__init__()

        self.gcn = GATNet(net_params=[n_in, 512, n_h], graph=graph)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.disc_c = Discriminator_cluster(n_h,n_h,n_nb,num_clusters)
        self.beta = beta
        self.cluster = Clusterator(n_h,num_clusters)
        
    def forward(self, seq1, seq2, g, sparse, msk, samp_bias1, samp_bias2, cluster_temp):
        
        h_1 = self.gcn(g, seq1, 0, 0)
        h_2 = self.gcn(g, seq2, 0, 0)
        self.beta = cluster_temp

        Z, S = self.cluster(h_1[-1,:,:], cluster_temp)
        Z_t = S @ Z
        c2 = Z_t
        
        c2 = self.sigm(c2)
        
        c = self.read(h_1, msk)
        c = self.sigm(c) 
        c_x = c.unsqueeze(1)
        c_x = c_x.expand_as(h_1)
        
        ret = self.disc(c_x, h_1, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc_c(c2, c2,h_1[-1,:,:], h_1[-1,:,:] ,h_2[-1,:,:], S , samp_bias1, samp_bias2)
        return ret, ret2 

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk, cluster_temp):
        
        h_1 = self.gcn(adj, seq, 0, 0)
        c = self.read(h_1, msk)
        Z, S = self.cluster(h_1[-1,:,:], self.beta)
        H = S@Z
        return h_1.detach(), H.detach(), c.detach(), Z.detach()