# A Unified Lottery Tickets Hypothesis for Graph Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[Preprint] [A Unified Lottery Tickets Hypothesis for Graph Neural Networks]()

Tianlong Chen\*, Yongduo Sui\*, Xuxi Chen, Aston Zhang, Zhangyang Wang

## Overview

<img src = "./Figs/Teaser.png" align = "left" width="50%" hight="50%"> Summary of our achieved performance (y-axis) at different graph and GNN sparsity levels (x-axis) on Cora and Citeceer node classification. The size of markers represent the inference MACs ($=\frac{1}{2}$ FLOPs) of each sparse GCN on the corresponding sparsified graphs. Black circles ($\bullet$) indicate the baseline, i.e., unpruned dense GNNs on the full graph. Blue circles ($\textcolor{blue}{\bullet}$) are random pruning results. Orange circles ($\textcolor{orange}{\bullet}$) represent  the performance of a previous graph sparsification approach, i.e., ADMM. Red stars  are established by our method (UGS).

## Methodlody

https://github.com/Shen-Lab/SS-GCNs

https://github.com/cmavro/Graph-InfoClust-GIC

https://github.com/lightaime/deep_gcns_torch

## 2. Experiments

### 2.1 Node classification on Cora, Citeseer, PubMed

```
cd NodeClassification
```

### 2.2 Link Prediction on Cora, Citeseer, PubMed

```
cd LinkPrediction
```

### 2.3 Experiments on OGB datasets

```
cd OGBN_arxiv/unify/ogb/ogbn_arxiv

cd OGBN_proteins/unify/ogb/ogbn_proteins

cd OGBL_Collab/unify/ogb/ogbl_collab

```


