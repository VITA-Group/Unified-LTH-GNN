from utils.data_util import intersection, process_indexes
from utils.ckpt_util import save_ckpt
import pdb
import numpy as np
import torch
import torch.optim as optim
import statistics
import pruning


def train_baseline(data, dataset, model, optimizer, criterion, device, args):

    loss_list = []
    model.train()
    sg_nodes, sg_edges, sg_edges_index, _ = data

    train_y = dataset.y[dataset.train_idx]
    idx_clusters = np.arange(len(sg_nodes))
    np.random.shuffle(idx_clusters)
    
    for idx in idx_clusters:

        x = dataset.x[sg_nodes[idx]].float().to(device)
        sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).to(device)

        sg_edges_ = sg_edges[idx].to(device)
        sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].to(device)

        mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}
        inter_idx = intersection(sg_nodes[idx], dataset.train_idx.tolist())
        training_idx = [mapper[t_idx] for t_idx in inter_idx]

        for iteration in range(args.iteration):
            optimizer.zero_grad()
            pred = model(x, sg_nodes_idx, sg_edges_, sg_edges_attr)
            target = train_y[inter_idx].to(device)
            loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
    return statistics.mean(loss_list)


def train_mask(epoch, data, dataset, model, optimizer, criterion, device, args):

    loss_list = []
    model.train()
    # subgraph node, subgraph edges
    '''
    sg_nodes: [real node index] * 10
    sg_edges: [local node index - local node index] * 10
    sg_edges_index: [real edge index] * 10
    train_y: train label: [86619, 112]
    '''
    sg_nodes, sg_edges, sg_edges_index, sg_edges_orig, sg_edges_mask = data
    train_y = dataset.y[dataset.train_idx]
    idx_clusters = np.arange(len(sg_nodes)) # 10
    np.random.shuffle(idx_clusters)

    pruning_adj_num_per_subgraph = int(args.pruning_percent_adj * model.num_original_edge / args.epochs / len(sg_nodes))
    pruning_weight_percent = args.pruning_percent_wei / args.epochs

    for it, idx in enumerate(idx_clusters):

        x = dataset.x[sg_nodes[idx]].float().to(device) # get subgraph node features from sg_nodes[idx]
        sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).to(device) # 13183 real node index

        sg_edges_ = sg_edges[idx].to(device) # subgraph adj_matrix corsponding edge index  index
        sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].to(device) # get subgraph node features from real sg_edges_index
        sg_edges_mask_idx = sg_edges_mask[idx].to(device) # get edge mask
        sg_edges_orig_idx = sg_edges_orig[idx]

        mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}

        inter_idx = intersection(sg_nodes[idx], dataset.train_idx.tolist()) # 8558 node for training
        training_idx = [mapper[t_idx] for t_idx in inter_idx] # 8558 fpr training
        
        for iteration in range(args.iteration):
            optimizer.zero_grad()
            if iteration == 0:
                pred = model(x, sg_nodes_idx, sg_edges_, sg_edges_attr, sg_edges_mask_idx)
                optimizer.param_groups[0]['params'].append(model.edge_mask)
            else:
                pred = model(x, sg_nodes_idx, sg_edges_, sg_edges_attr)
            target = train_y[inter_idx].to(device)
            loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))
            loss.backward()
            pruning.subgradient_update_mask(model, args) # l1 norm
            optimizer.step()

        optimizer.param_groups[0]['params'].pop()
        
        loss_list.append(loss.item())
        pruning_links = pruning.pruning_real_adj_matrix(it, model, dataset, 
                                        sg_edges_orig_idx, 
                                        pruning_adj_num_per_subgraph)
        print("Epoch:[{}] | Subgraph:[{}/10] Pruning:[{}] links, Remain:[{}/{}={:.3f}%]"
            .format(epoch, it + 1, pruning_links, 
                    dataset.adj.nnz, 
                    model.num_original_edge, 
                    dataset.adj.nnz * 100 / model.num_original_edge))
        pruning.remove_mask(model)
    
    wei_remain, wei_nonzero = pruning.pruning_weight_mask(model, pruning_weight_percent)
    adj_spar = dataset.adj.nnz / model.num_original_edge
    wei_spar = wei_remain / wei_nonzero
    return statistics.mean(loss_list),  adj_spar, wei_spar.cpu().item()


def train_fixed(data, dataset, model, optimizer, criterion, device, args):

    loss_list = []
    model.train()
    # subgraph node, subgraph edges
    '''
    sg_nodes: [real node index] * 10
    sg_edges: [local node index - local node index] * 10
    sg_edges_index: [real edge index] * 10
    train_y: train label: [86619, 112]
    '''
    sg_nodes, sg_edges, sg_edges_index, _ , sg_edges_mask = data
    train_y = dataset.y[dataset.train_idx]
    idx_clusters = np.arange(len(sg_nodes)) # 10
    np.random.shuffle(idx_clusters)

    for idx in idx_clusters:

        x = dataset.x[sg_nodes[idx]].float().to(device) # get subgraph node features from sg_nodes[idx]
        sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).to(device) # 13183 real node index

        sg_edges_ = sg_edges[idx].to(device) # subgraph adj_matrix corsponding edge index
        sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].to(device) # get subgraph node features from real sg_edges_index
        sg_edges_mask_idx = sg_edges_mask[idx].to(device) # get edge mask

        mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}

        inter_idx = intersection(sg_nodes[idx], dataset.train_idx.tolist()) # 8558 node for training
        training_idx = [mapper[t_idx] for t_idx in inter_idx] # 8558 fpr training

        for iteration in range(args.iteration):
            optimizer.zero_grad()
            pred = model(x, sg_nodes_idx, sg_edges_, sg_edges_attr)
            target = train_y[inter_idx].to(device)
            loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))
            loss.backward()
            optimizer.step()

        # optimizer.zero_grad()
        # pred = model(x, sg_nodes_idx, sg_edges_, sg_edges_attr, sg_edges_mask_idx)
        # target = train_y[inter_idx].to(device)
        # loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))
        # loss.backward()
        # optimizer.step()
        loss_list.append(loss.item())

    return statistics.mean(loss_list)


@torch.no_grad()
def multi_evaluate(valid_data_list, dataset, model, evaluator, device):
    model.eval()
    target = dataset.y.detach().numpy()

    train_pre_ordered_list = []
    valid_pre_ordered_list = []
    test_pre_ordered_list = []

    test_idx = dataset.test_idx.tolist()
    train_idx = dataset.train_idx.tolist()
    valid_idx = dataset.valid_idx.tolist()

    for valid_data_item in valid_data_list:
        sg_nodes, sg_edges, sg_edges_index, _ = valid_data_item
        idx_clusters = np.arange(len(sg_nodes))

        test_predict = []
        test_target_idx = []

        train_predict = []
        valid_predict = []

        train_target_idx = []
        valid_target_idx = []

        for idx in idx_clusters:
            x = dataset.x[sg_nodes[idx]].float().to(device)
            sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).to(device)

            mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}
            sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].to(device)

            inter_tr_idx = intersection(sg_nodes[idx], train_idx)
            inter_v_idx = intersection(sg_nodes[idx], valid_idx)

            train_target_idx += inter_tr_idx
            valid_target_idx += inter_v_idx

            tr_idx = [mapper[tr_idx] for tr_idx in inter_tr_idx]
            v_idx = [mapper[v_idx] for v_idx in inter_v_idx]

            pred = model(x, sg_nodes_idx, sg_edges[idx].to(device), sg_edges_attr).cpu().detach()

            train_predict.append(pred[tr_idx])
            valid_predict.append(pred[v_idx])

            inter_te_idx = intersection(sg_nodes[idx], test_idx)
            test_target_idx += inter_te_idx

            te_idx = [mapper[te_idx] for te_idx in inter_te_idx]
            test_predict.append(pred[te_idx])

        train_pre = torch.cat(train_predict, 0).numpy()
        valid_pre = torch.cat(valid_predict, 0).numpy()
        test_pre = torch.cat(test_predict, 0).numpy()

        train_pre_ordered = train_pre[process_indexes(train_target_idx)]
        valid_pre_ordered = valid_pre[process_indexes(valid_target_idx)]
        test_pre_ordered = test_pre[process_indexes(test_target_idx)]

        train_pre_ordered_list.append(train_pre_ordered)
        valid_pre_ordered_list.append(valid_pre_ordered)
        test_pre_ordered_list.append(test_pre_ordered)

    train_pre_final = torch.mean(torch.Tensor(train_pre_ordered_list), dim=0)
    valid_pre_final = torch.mean(torch.Tensor(valid_pre_ordered_list), dim=0)
    test_pre_final = torch.mean(torch.Tensor(test_pre_ordered_list), dim=0)

    eval_result = {}

    input_dict = {"y_true": target[train_idx], "y_pred": train_pre_final}
    eval_result["train"] = evaluator.eval(input_dict)

    input_dict = {"y_true": target[valid_idx], "y_pred": valid_pre_final}
    eval_result["valid"] = evaluator.eval(input_dict)

    input_dict = {"y_true": target[test_idx], "y_pred": test_pre_final}
    eval_result["test"] = evaluator.eval(input_dict)

    return eval_result