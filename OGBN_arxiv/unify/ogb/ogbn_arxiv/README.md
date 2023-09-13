# UGS Node Classification on Ogbn-ArXiv
## 1. Requirements

```
conda create -n deepgcn python=3.7
source activate deepgcn
# make sure pytorch version >=1.4.0
conda install -y pytorch=1.4.0 torchvision cudatoolkit=10.1 python=3.7 -c pytorch
pip install torch==1.4.0 torchvision==0.5.0
pip install tensorboard

# command to install pytorch geometric, please refer to the official website for latest installation.
#  https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
CUDA=cu101
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install torch-sparse==0.6.1 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install torch-cluster==1.4.5 -f https://pytorch-geometric.com/whl/torch-1.4.0+${CUDA}.html
pip install torch-geometric==1.4.3
pip install requests

# install useful modules
pip install tqdm
pip install ogb

```


## 2. Training & Evaluation

### UGS and RP 

```
python main_imp.py \
--use_gpu \
--self_loop \
--learn_t \
--num_layers 28 \
--block res+ \
--gcn_aggr softmax_sg \
--t 0.1 \
--s1 1e-6 \
--s2 1e-4 \
--pruning_percent_wei 0.2 \
--pruning_percent_adj 0.05 \
--mask_epochs 250 \
--fix_epochs 500 \
--model_save_path IMP


python -u main_rp.py \
--use_gpu \
--self_loop \
--num_layers 28 \
--block res+ \
--gcn_aggr softmax_sg \
--t 0.1 \
--epochs 500 \
--model_save_path RP

```


