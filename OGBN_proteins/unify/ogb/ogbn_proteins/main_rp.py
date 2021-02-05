import __init__
import torch
import torch.optim as optim
import statistics
from dataset import OGBNDataset
from model import DeeperGCN
from args import ArgsInit
import time
import numpy as np
from ogb.nodeproppred import Evaluator
from utils.ckpt_util import save_ckpt
from utils.data_util import intersection, process_indexes
import logging
import pdb
import train
import pruning
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main_fixed_mask(args, imp_num, resume_train_ckpt=None):

    device = torch.device("cuda:" + str(args.device))
    dataset = OGBNDataset(dataset_name=args.dataset)
    nf_path = dataset.extract_node_features(args.aggr)

    args.num_tasks = dataset.num_tasks
    args.nf_path = nf_path

    evaluator = Evaluator(args.dataset)
    criterion = torch.nn.BCEWithLogitsLoss()

    valid_data_list = []
    for i in range(args.num_evals):

        parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=args.valid_cluster_number)
        valid_data = dataset.generate_sub_graphs(parts, cluster_number=args.valid_cluster_number)
        valid_data_list.append(valid_data)

    print("-" * 120)
    model = DeeperGCN(args).to(device)
    pruning.add_mask(model)
    pruning.random_pruning(dataset, model, args)
    adj_spar, wei_spar = pruning.print_sparsity(dataset, model)

    for name, param in model.named_parameters():
        if 'mask' in name:
            param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    results = {'highest_valid': 0, 'final_train': 0, 'final_test': 0, 'highest_train': 0, 'epoch':0}
    results['adj_spar'] = adj_spar
    results['wei_spar'] = wei_spar
    
    start_epoch = 1
    if resume_train_ckpt:
        dataset.adj = resume_train_ckpt['adj']
        start_epoch = resume_train_ckpt['epoch']
        rewind_weight_mask = resume_train_ckpt['rewind_weight_mask']
        ori_model_dict = model.state_dict()
        over_lap = {k : v for k, v in resume_train_ckpt['model_state_dict'].items() if k in ori_model_dict.keys()}
        ori_model_dict.update(over_lap)
        model.load_state_dict(ori_model_dict)
        print("Resume at RP:[{}] epoch:[{}] len:[{}/{}]!".format(imp_num, resume_train_ckpt['epoch'], len(over_lap.keys()), len(ori_model_dict.keys())))
        optimizer.load_state_dict(resume_train_ckpt['optimizer_state_dict'])
        adj_spar, wei_spar = pruning.print_sparsity(dataset, model)
    
    for epoch in range(start_epoch, args.epochs + 1):
        # do random partition every epoch
        t0 = time.time()
        train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=args.cluster_number)
        data = dataset.generate_sub_graphs(train_parts, cluster_number=args.cluster_number, ifmask=True)
        epoch_loss = train.train_fixed(data, dataset, model, optimizer, criterion, device, args)
        result = train.multi_evaluate(valid_data_list, dataset, model, evaluator, device)

        train_result = result['train']['rocauc']
        valid_result = result['valid']['rocauc']
        test_result = result['test']['rocauc']

        if valid_result > results['highest_valid']:
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result
            results['epoch'] = epoch
            final_state_dict = pruning.save_all(dataset, 
                                                model, 
                                                None, 
                                                optimizer, 
                                                imp_num, 
                                                epoch, 
                                                args.model_save_path, 
                                                'RP{}_fixed_ckpt'.format(imp_num))
        epoch_time = (time.time() - t0) / 60
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
              'RP:[{}] (FIX Mask) Epoch[{}/{}] LOSS[{:.4f}] Train[{:.2f}] Valid[{:.2f}] Test[{:.2f}] | Update Test[{:.2f}] at epoch[{}] | Adj[{:.2f}%] Wei[{:.2f}%] Time[{:.2f}min]'
              .format(imp_num, epoch, args.epochs, epoch_loss, train_result * 100,
                                                               valid_result * 100,
                                                               test_result * 100,
                                                               results['final_test'] * 100,
                                                               results['epoch'],
                                                               results['adj_spar'],
                                                               results['wei_spar'],
                                                               epoch_time))
    print("=" * 120)
    print("syd final: RP:[{}], Train:[{:.2f}]  Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] | Adj:[{:.2f}%] Wei:[{:.2f}%]"
        .format(imp_num,    results['final_train'] * 100,
                            results['highest_valid'] * 100,
                            results['epoch'],
                            results['final_test'] * 100,
                            results['adj_spar'] * 100,
                            results['wei_spar'] * 100))
    print("=" * 120)


if __name__ == "__main__":

    args = ArgsInit().save_exp()
    pruning.setup_seed(666)
    imp_num = args.imp_num

    percent_list = [(1 - (1 - 0.05) ** (i + 1), 1 - (1 - 0.2) ** (i + 1)) for i in range(20)]
    args.pruning_percent_adj, args.pruning_percent_wei = percent_list[imp_num - 1]
    pruning.print_args(args, 80)
    print("syd: RP:[{}] Pruning adj[{:.6f}], wei[{:.6f}]".format(imp_num, args.pruning_percent_adj, args.pruning_percent_wei))

    resume_train_ckpt = None
    
    if args.resume_dir:
        resume_train_ckpt = torch.load(args.resume_dir)
        imp_num = resume_train_ckpt['imp_num']
        
    main_fixed_mask(args, imp_num, resume_train_ckpt)

