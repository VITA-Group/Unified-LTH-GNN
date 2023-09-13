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
import train
import pruning
import pdb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
    
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

    model = DeeperGCN(args).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    results = {'highest_valid': 0, 'final_train': 0, 'final_test': 0, 'highest_train': 0, 'epoch':0}

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        # do random partition every epoch
        train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=args.cluster_number)
        data = dataset.generate_sub_graphs(train_parts, cluster_number=args.cluster_number)

        epoch_loss = train.train_baseline(data, dataset, model, optimizer, criterion, device, args)
        result = train.multi_evaluate(valid_data_list, dataset, model, evaluator, device)
        train_result = result['train']['rocauc']
        valid_result = result['valid']['rocauc']
        test_result = result['test']['rocauc']

        if valid_result > results['highest_valid']:
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result
            results['epoch'] = epoch
            save_dir = save_ckpt(model, optimizer, round(epoch_loss, 4),
                      epoch,
                      args.model_save_path,
                      name_post='valid_best')

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
              '(Baseline) Epoch:[{}/{}] LOSS:[{:.4f}] Train :[{:.2f}] Valid:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at epoch:[{}]'
              .format(epoch, args.epochs, epoch_loss, train_result * 100,
                                                      valid_result * 100,
                                                      test_result * 100,
                                                      results['final_test'] * 100,
                                                      results['epoch']))
    end_time = time.time()
    total_time = end_time - start_time
    # logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))
    print("=" * 120)
    print("syd final: BASELINE, Train:[{:.2f}]  Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] Total time:[{}]"
        .format(results['final_train'] * 100,
                results['highest_valid'] * 100,
                results['epoch'],
                results['final_test'] * 100,
                time.strftime('%H:%M:%S', time.gmtime(total_time))))
    print("=" * 120)


if __name__ == "__main__":

    args = ArgsInit().save_exp()
    pruning.print_args(args, 80)
    pruning.setup_seed(666)
    main(args)
