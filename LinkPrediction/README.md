# UGS Link Prediction on Cora, Citeseer, PubMed Datasets
## 1. Requirements

```
python==3.6

pytorch >= 1.4.0

dgl-cu101==0.4.2
```


## 2. Training & Evaluation

### GCN IMP, RP and Pretrain

```
python main_gcn_imp.py --dataset cora --mask_epoch 200 --fix_epoch 200 --s1 1e-2 --s2 1e-2

python main_gcn_imp.py --dataset citeseer --mask_epoch 200 --fix_epoch 200 --s1 1e-2 --s2 1e-2

python main_gcn_imp.py --dataset pubmed --mask_epoch 200 --fix_epoch 200 --s1 1e-2 --s2 1e-2

python main_gcn_rp.py --dataset cora --fix_epoch 200

python main_gcn_rp.py --dataset citeseer --fix_epoch 200

python main_gcn_rp.py --dataset pubmed --fix_epoch 200

python -u main_gcn_imp_pretrain.py --dataset cora --mask_epoch 200 --fix_epoch 200 --s1 1e-2 --s2 1e-2 --weight_dir cora_double_dgi.pkl

python -u main_gcn_imp_pretrain.py --dataset citeseer --mask_epoch 200 --fix_epoch 200 --s1 1e-2 --s2 1e-2 --weight_dir cite_double_dgi.pkl


```


### GAT & GIN IMP and RP

```
python main_gingat_imp.py --net gin --dataset cora --mask_epoch 200 --fix_epoch 200--s1 1e-2 --s2 1e-2

python main_gingat_imp.py --net gin --dataset citeseer --mask_epoch 200 --fix_epoch 200--s1 1e-2 --s2 1e-2

python main_gingat_imp.py --net gin --dataset pubmed --mask_epoch 200 --fix_epoch 200--s1 1e-2 --s2 1e-2

python main_gingat_rp.py --net gin --dataset cora --fix_epoch 200

python main_gingat_rp.py --net gin --dataset citeseer --fix_epoch 200

python main_gingat_rp.py --net gin --dataset pubmed --fix_epoch 200

python main_gingat_imp.py --net gat --dataset cora --mask_epoch 200 --fix_epoch 200--s1 1e-2 --s2 1e-2

python main_gingat_imp.py --net gat --dataset citeseer --mask_epoch 200 --fix_epoch 200--s1 1e-2 --s2 1e-2

python main_gingat_imp.py --net gat --dataset pubmed --mask_epoch 200 --fix_epoch 200--s1 1e-2 --s2 1e-2

python main_gingat_rp.py --net gat --dataset cora --fix_epoch 200

python main_gingat_rp.py --net gat --dataset citeseer --fix_epoch 200

python main_gingat_rp.py --net gat --dataset pubmed --fix_epoch 200

```