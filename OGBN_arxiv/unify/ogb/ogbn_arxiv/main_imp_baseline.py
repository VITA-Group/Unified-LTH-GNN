import __init__
from ogb.nodeproppred import Evaluator
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, add_self_loops
from args import ArgsInit
from ogb.nodeproppred import PygNodePropPredDataset
from model import DeeperGCN
from utils.ckpt_util import save_ckpt
import logging
import pruning
import copy
import time
import pdb
import warnings
warnings.filterwarnings('ignore')

@torch.no_grad()
def test(model, x, edge_index, y_true, split_idx, evaluator):
    model.eval()
    out = model(x, edge_index)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def train_fixed(model, x, edge_index, y_true, train_idx, optimizer, args):

    model.train()
    optimizer.zero_grad()
    pred = model(x, edge_index)[train_idx]
    loss = F.nll_loss(pred, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


def main_fixed_mask(args):

    device = torch.device("cuda:" + str(args.device))
    dataset = PygNodePropPredDataset(name=args.dataset)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset)

    x = data.x.to(device)
    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)

    edge_index = data.edge_index.to(device)
    edge_index = to_undirected(edge_index, data.num_nodes)

    if args.self_loop:
        edge_index = add_self_loops(edge_index, num_nodes=data.num_nodes)[0]

    args.in_channels = data.x.size(-1)
    args.num_tasks = dataset.num_classes

    model = DeeperGCN(args).to(device)
    pruning.add_mask(model, args.num_layers)
    
    for name, param in model.named_parameters():
        if 'mask' in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    results = {'highest_valid': 0, 'final_train': 0, 'final_test': 0, 'highest_train': 0, 'epoch': 0}
    
    start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
    
        epoch_loss = train_fixed(model, x, edge_index, y_true, train_idx, optimizer, args)
        result = test(model, x, edge_index, y_true, split_idx, evaluator)
        train_accuracy, valid_accuracy, test_accuracy = result

        if valid_accuracy > results['highest_valid']:
            results['highest_valid'] = valid_accuracy
            results['final_train'] = train_accuracy
            results['final_test'] = test_accuracy
            results['epoch'] = epoch

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
              'Baseline (FIX Mask) Epoch:[{}/{}]\t LOSS:[{:.4f}] Train :[{:.2f}] Valid:[{:.2f}] Test:[{:.2f}] | Update Test:[{:.2f}] at epoch:[{}]'
              .format(epoch, args.epochs, epoch_loss, train_accuracy * 100,
                                                               valid_accuracy * 100,
                                                               test_accuracy * 100, 
                                                               results['final_test'] * 100,
                                                               results['epoch']))
    print("=" * 120)
    print("syd final: Baseline, Train:[{:.2f}]  Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}]"
        .format(            results['final_train'] * 100,
                            results['highest_valid'] * 100,
                            results['epoch'],
                            results['final_test'] * 100))
    print("=" * 120)


if __name__ == "__main__":

    args = ArgsInit().save_exp()
    pruning.print_args(args, 120)
    pruning.setup_seed(args.seed)
    main_fixed_mask(args)

        
