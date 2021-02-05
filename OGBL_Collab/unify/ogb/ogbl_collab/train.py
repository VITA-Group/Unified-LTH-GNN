import torch
import pdb
from torch.utils.data import DataLoader
import pruning

@torch.no_grad()
def test(model, predictor, x, edge_index, split_edge, evaluator, batch_size, args):
    model.eval()
    predictor.eval()

    h = model(x, edge_index)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def train_fixed(model, predictor, x, edge_index, split_edge, optimizer, batch_size, args):

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0

    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):

        optimizer.zero_grad()
        h = model(x, edge_index)
        # positive edges
        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        # add a extremely small value to avoid gradient explode
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        # negative edges
        edge = torch.randint(0, x.size(0),
                             edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        # why need to do this? clip grad norm
        # https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48
        # ||g|| < c if not new_g = g / ||g||
        # tackle exploding gradients issue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples



def train_mask(model, predictor, x, edge_index, split_edge, optimizer, args):

    model.train()
    predictor.train()
    
    pos_train_edge = split_edge['train']['edge'].to(x.device)
    total_loss = total_examples = 0

    for perm in DataLoader(range(pos_train_edge.size(0)), args.batch_size, shuffle=True):
        
        optimizer.zero_grad()
        h = model(x, edge_index)
        # positive edges
        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        # add a extremely small value to avoid gradient explode
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        # negative edges
        edge = torch.randint(0, x.size(0),
                             edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        # why need to do this? clip grad norm
        # https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48
        # ||g|| < c if not new_g = g / ||g||
        # tackle exploding gradients issue
        pruning.subgradient_update_mask(model, args) # l1 norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    prune_info_dict = pruning.pruning_mask(model, args)

    # print('pruning: adj:[{}/{}={}]'.format(prune_info_dict['adj_prune'], prune_info_dict['adj_total'], prune_info_dict['adj_spar']))
    # print('pruning: wei:[{}/{}={}]'.format(prune_info_dict['wei_prune'], prune_info_dict['wei_total'], prune_info_dict['wei_spar']))
    return total_loss / total_examples, prune_info_dict