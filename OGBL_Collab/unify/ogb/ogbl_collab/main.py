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

def main(args):

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
    predictor = LinkPredictor(args).to(device)

    logging.info(model)
    logging.info(predictor)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)

    results = {}
    keys = ['highest_valid', 'final_train', 'final_test', 'highest_train']
    hits = ['Hits@10', 'Hits@50', 'Hits@100']

    for key in keys:
        results[key] = {k: 0 for k in hits}

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):

        epoch_loss = train.train(model, predictor,
                           x, edge_index,
                           split_edge,
                           optimizer, args.batch_size)
        logging.info('Epoch {}, training loss {:.4f}'.format(epoch, epoch_loss))
        result = train.test(model, predictor,
                      x, edge_index,
                      split_edge,
                      evaluator, args.batch_size)

        for k in hits:
            # return a tuple
            train_result, valid_result, test_result = result[k]

            if train_result > results['highest_train'][k]:
                results['highest_train'][k] = train_result

            if valid_result > results['highest_valid'][k]:
                results['highest_valid'][k] = valid_result
                results['final_train'][k] = train_result
                results['final_test'][k] = test_result

                save_ckpt(model, optimizer,
                          round(epoch_loss, 4), epoch,
                          args.model_save_path,
                          k, name_post='valid_best')
                save_ckpt(predictor, optimizer,
                          round(epoch_loss, 4), epoch,
                          args.model_save_path,
                          k, name_post='valid_best_link_predictor')

        logging.info(result)

    logging.info("%s" % results)

    end_time = time.time()
    total_time = end_time - start_time
    time_used = 'Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time)))
    logging.info(time_used)


if __name__ == "__main__":

    args = ArgsInit().save_exp()
    pruning.print_args(args, 80)
    pruning.setup_seed(666)

    main(args)
