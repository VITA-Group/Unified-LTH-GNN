# UGS Node Classification on Ogbn-Proteins
## 1. Requirements

```
conda create -n deepgcn python=3.7
source activate deepgcn
# make sure pytorch version >=1.4.0
# conda install -y pytorch=1.4.0 torchvision cudatoolkit=10.0 python=3.7 -c pytorch
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
python -u main_imp.py \
--use_gpu \
--conv_encode_edge \
--use_one_hot_encoding \
--learn_t \
--num_layers 28 \
--s1 1e-1 \
--s2 1e-3 \
--epochs 100 \
--iteration 10 \
--model_save_path IMP \
--imp_num 1


python -u main_rp.py \
--use_gpu \
--conv_encode_edge \
--use_one_hot_encoding \
--learn_t \
--num_layers 28 \
--epochs 100 \
--iteration 10 \
--model_save_path RP \
--imp_num 1

```


