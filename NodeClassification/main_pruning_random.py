import os
import random
import argparse

import torch
import torch.nn as nn
import numpy as np
import csv

import net as net
from utils import load_data
from sklearn.metrics import f1_score
import pdb
import pruning
import copy
from scipy.sparse import coo_matrix
import warnings
import time
from v2GNNAccel.utils.experiments_utils import check_and_create_csv

warnings.filterwarnings("ignore")

PREPROCESSING_TIME = 0


# python main_pruning_random.py --dataset grph_6 --dataset_path /home/polp/puigde/v2GNNAccel/datasets/minisample/lt --embedding-dim 413 512 19 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --total_epoch 200
# python main_pruning_random.py --dataset cora --dataset_path ../dataset --embedding-dim 1433 512 7 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --total_epoch 200
def run_fix_mask(args, seed, adj_percent, wei_percent):
    pruning.setup_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data(
        args["dataset"], args["dataset_path"]
    )

    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()

    net_gcn = net.net_gcn(embedding_dim=args["embedding_dim"], adj=adj)
    pruning.add_mask(net_gcn)
    net_gcn = net_gcn.cuda()
    preprocessing_start = time.perf_counter()
    pruning.random_pruning(net_gcn, adj_percent, wei_percent)
    PREPROCESSING_TIME = time.perf_counter() - preprocessing_start

    adj_spar, wei_spar = pruning.print_sparsity(net_gcn)

    for name, param in net_gcn.named_parameters():
        if "mask" in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        net_gcn.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )
    acc_test = 0.0
    best_val_acc = {"val_acc": 0, "epoch": 0, "test_acc": 0}
    total_time = 0
    for epoch in range(args["total_epoch"]):
        start_time = time.time()
        optimizer.zero_grad()
        output = net_gcn(features, adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        end_time = time.time()
        total_time += end_time - start_time
        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
            acc_val = f1_score(
                labels[idx_val].cpu().numpy(),
                output[idx_val].cpu().numpy().argmax(axis=1),
                average="micro",
            )
            acc_test = f1_score(
                labels[idx_test].cpu().numpy(),
                output[idx_test].cpu().numpy().argmax(axis=1),
                average="micro",
            )
            if acc_val > best_val_acc["val_acc"]:
                best_val_acc["val_acc"] = acc_val
                best_val_acc["test_acc"] = acc_test
                best_val_acc["epoch"] = epoch
        print(
            "(Fix Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]".format(
                epoch,
                acc_val * 100,
                acc_test * 100,
                best_val_acc["val_acc"] * 100,
                best_val_acc["test_acc"] * 100,
                best_val_acc["epoch"],
            )
        )

    return (
        best_val_acc["val_acc"],
        best_val_acc["test_acc"],
        best_val_acc["epoch"],
        adj_spar,
        wei_spar,
        total_time * 1e3 / args["total_epoch"],
    )


def parser_loader():
    parser = argparse.ArgumentParser(description="GLT")
    ###### Unify pruning settings #######
    parser.add_argument(
        "--s1", type=float, default=0.0001, help="scale sparse rate (default: 0.0001)"
    )
    parser.add_argument(
        "--s2", type=float, default=0.0001, help="scale sparse rate (default: 0.0001)"
    )
    parser.add_argument("--total_epoch", type=int, default=300)
    parser.add_argument("--pruning_percent", type=float, default=0.1)
    parser.add_argument("--pruning_percent_wei", type=float, default=0.1)
    parser.add_argument("--pruning_percent_adj", type=float, default=0.1)
    parser.add_argument("--weight_dir", type=str, default="")
    ###### Others settings #######
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--embedding-dim", nargs="+", type=int, default=[3703, 16, 6])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--csv_filename", type=str, help="csv filename and path")
    parser.add_argument("--graph_id", type=str)
    return parser


if __name__ == "__main__":
    parser = parser_loader()
    args = vars(parser.parse_args())

    seed_dict = {"cora": 3846, "citeseer": 2839, "pubmed": 3333}
    seed = seed_dict.get(args["dataset"], 3333)

    percent_list = [
        (
            1 - (1 - args["pruning_percent_adj"]) ** (i + 1),
            1 - (1 - args["pruning_percent_wei"]) ** (i + 1),
        )
        for i in range(20)
    ]
    for p, (adj_percent, wei_percent) in enumerate(percent_list):
        (
            best_acc_val,
            final_acc_test,
            final_epoch_list,
            adj_spar,
            wei_spar,
            avg_epoch_time,
        ) = run_fix_mask(args, seed, adj_percent, wei_percent)
        print("=" * 120)
        print(
            "syd : Sparsity:[{}], Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]".format(
                p + 1,
                best_acc_val * 100,
                final_epoch_list,
                final_acc_test * 100,
                adj_spar,
                wei_spar,
            )
        )
        print("=" * 120)

    ed = args["embedding_dim"]
    node_features = ed[0]
    num_classes = ed[-1]
    hid = -1
    if len(ed) > 2:
        hid = ed[1]
    num_layers = len(ed)
    check_and_create_csv(
        dataset=args["dataset_path"],
        dim=node_features,
        hidden=hid,
        classes=num_classes,
        num_layers=num_layers,
        model="gcn",
        avg_epoch_time=avg_epoch_time,
        best_accuracy=final_acc_test,
        epoch_best_accuracy=final_epoch_list,
        csv_filename=args["csv_filename"],
        graph_id=args["graph_id"],
        epochs=args["total_epoch"],
        preprocessing_time=PREPROCESSING_TIME * 1e3,
    )
