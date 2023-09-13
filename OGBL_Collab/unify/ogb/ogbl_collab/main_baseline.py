import __init__
from ogb.nodeproppred import Evaluator
import torch
from torch.utils.data import DataLoader
from args import ArgsInit
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from model import DeeperGCN, LinkPredictor
from utils.ckpt_util import save_ckpt
import logging
import time
from torch.utils.tensorboard import SummaryWriter
import train
import pdb
import pruning
import copy


def main_get_mask(args, imp_num):

    device = torch.device("cuda:" + str(args.device))
    dataset = PygLinkPropPredDataset(name=args.dataset)
    data = dataset[0]
    
    # Data(edge_index=[2, 2358104], edge_weight=[2358104, 1], edge_year=[2358104, 1], x=[235868, 128])
    split_edge = dataset.get_edge_split()
    evaluator = Evaluator(args.dataset)

    x = data.x.to(device)

    edge_index = data.edge_index.to(device)

    args.in_channels = data.x.size(-1)
    args.num_tasks = 1

    model = DeeperGCN(args).to(device)
    pruning.add_mask(model, args)

    for name, param in model.named_parameters():
        if 'mask' in name:
            param.requires_grad = False

    predictor = LinkPredictor(args).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)

    results = {'epoch':0}
    keys = ['highest_valid', 'final_train', 'final_test', 'highest_train']
    hits = ['Hits@10', 'Hits@50', 'Hits@100']

    for key in keys:
        results[key] = {k: 0 for k in hits}

    start_epoch = 1
    for epoch in range(start_epoch, args.mask_epochs + 1):

        t0 = time.time()
        
        epoch_loss = train.train_fixed(model, predictor, x, edge_index, split_edge, optimizer, args.batch_size, args)
        result = train.test(model, predictor, x, edge_index, split_edge, evaluator, args.batch_size, args)

        k = 'Hits@50'
        train_result, valid_result, test_result = result[k]

        if train_result > results['highest_train'][k]:
            results['highest_train'][k] = train_result

        if valid_result > results['highest_valid'][k]:
            results['highest_valid'][k] = valid_result
            results['final_train'][k] = train_result
            results['final_test'][k] = test_result
            results['epoch'] = epoch
            
        epoch_time = (time.time() - t0) / 60
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
              'IMP:[{}] (GET Mask) Epoch:[{}/{}] LOSS:[{:.4f}] Train :[{:.2f}] Valid:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at epoch:[{}] Time:[{:.2f}min]'
              .format(imp_num, epoch, args.mask_epochs, epoch_loss, train_result * 100,
                                                               valid_result * 100,
                                                               test_result * 100,
                                                               results['final_test'][k] * 100,
                                                               results['epoch'],
                                                               epoch_time))
    print('-' * 100)
    print("syd : IMP:[{}] (FIX Mask) Final Result Train:[{:.2f}]  Valid:[{:.2f}]  Test:[{:.2f}]"
        .format(imp_num, results['final_train'][k] * 100,
                         results['highest_valid'][k] * 100,
                         results['final_test'][k] * 100))
    print('-' * 100)



if __name__ == "__main__":

    args = ArgsInit().save_exp()
    pruning.print_args(args, 80)
    pruning.setup_seed(666)
    imp_num = 0
    main_get_mask(args, imp_num)
        