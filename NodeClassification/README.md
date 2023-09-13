# UGS Node Classification on Cora, Citeseer, PubMed Datasets
## 1. Requirements

```
python==3.6

pytorch >= 1.4.0

dgl-cu101==0.4.2
```


## 2. Training & Evaluation

Notes: for retraining stage, we use the fixed 200 epochs and we extract the mask at the epoch with the best val accuracy. We can also use the early stop strategy (according to the val acc) to speed up the retraining stage and avoid the performance drop.

### GCN IMP, RP and Pretrain

```
python main_pruning_imp.py --dataset cora --embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one

python main_pruning_imp.py --dataset citeseer --embedding-dim 3703 512 6 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --total_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one

python main_pruning_imp.py --dataset pubmed --embedding-dim 500 512 3 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --total_epoch 200 --s1 1e-6 --s2 1e-3 --init_soft_mask_type all_one

python main_pruning_random.py --dataset cora --embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --total_epoch 200

python main_pruning_random.py --dataset citeseer --embedding-dim 3703 512 6 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --total_epoch 200

python main_pruning_random.py --dataset pubmed ---embedding-dim 500 512 3 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --total_epoch 200

python -u main_pruning_imp_pretrain.py --dataset cora --embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --fix_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one --weight_dir cora_double_dgi.pkl

python -u main_pruning_imp_pretrain.py --dataset citeseer --embedding-dim 3703 512 6 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --fix_epoch 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one --weight_dir cite_double_dgi.pkl

```

### ADMM baseline

```
cd ADMM/ADMM

Run pretrain.py to obtain pretrained model 

Run train-auto-admm-tuneParameter.py to get the ADMM adjacency matrix

Then

python -u main_admm_eval.py --dataset cora --embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 --index $adj_index

to eval the ADMM baseline performance
```


### GAT & GIN IMP and RP

```

python -u main_gingat_imp.py --dataset cora --net gin --embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epoch 200 --fix_epoch 200 --s1 1e-3 --s2 1e-3

python -u main_gingat_imp.py --dataset citeseer --net gin --embedding-dim 3703 512 6 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epoch 200 --fix_epoch 200 --s1 1e-5 --s2 1e-5

python -u main_gingat_imp.py --dataset pubmed --net gin --embedding-dim 500 512 3 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epoch 200 --fix_epoch 200 --s1 1e-5 --s2 1e-5

python -u main_gingat_rp.py --dataset cora --net gin --embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --fix_epoch 200

python -u main_gingat_rp.py --dataset citeseer --net gin --embedding-dim 3703 512 6 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --fix_epoch 200

python -u main_gingat_rp.py --dataset pubmed --net gin --embedding-dim 500 512 3 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --fix_epoch 200


python -u main_gingat_imp.py --dataset cora --net gat --embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epoch 200 --fix_epoch 200 --s1 1e-3 --s2 1e-3

python -u main_gingat_imp.py --dataset citeseer --net gat --embedding-dim 3703 512 6 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epoch 200 --fix_epoch 200 --s1 1e-7 --s2 1e-3

python -u main_gingat_imp.py --dataset pubmed --net gat --embedding-dim 500 512 3 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epoch 200 --fix_epoch 200 --s1 1e-2 --s2 1e-2

python -u main_gingat_rp.py --dataset cora --net gat --embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --fix_epoch 200

python -u main_gingat_rp.py --dataset citeseer --net gat --embedding-dim 3703 512 6 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --fix_epoch 200

python -u main_gingat_rp.py --dataset pubmed --net gat --embedding-dim 500 512 3 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --fix_epoch 200
```
