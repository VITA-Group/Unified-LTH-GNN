import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf
import shutil

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("cora_10_fold/ind.{}.{}.0".format(dataset_str, names[i]), 'rb') as f:
        #10 folds
        #print("cora_10_fold/ind.{}.{}.0".format(dataset_str, names[i]))
        #with open("citeseer_10_fold/ind.{}.{}.1".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("cora_10_fold/ind.{}.test.index.0".format(dataset_str))
    #for 10 folds
    #test_idx_reorder = parse_index_file("citeseer_10_fold/ind.{}.test.index.1".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    #print("number of edge: ", nx.from_dict_of_lists(graph).number_of_edges())
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    #print("len of idx_test", len(idx_test))
    idx_train = range(len(y))
    #print("len of idx_train", len(idx_train))
    idx_val = range(len(y), len(y)+500)
    #print("len of idx_val", len(idx_val))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]


    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1), dtype=float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    #print("adj shape:", adj.toarray().shape)
    #print("number non zero original adj", np.count_nonzero(adj.toarray()))
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = adj + sp.eye(adj.shape[0])
    return adj_normalized.toarray()


def construct_feed_dict(features, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def partial_mask(adj):
    adj_np_array = np.array(adj.toarray(), dtype=np.float32)
    #adj_np_array = np.copy(adj)
    return adj_np_array


def all_adj_mask(adj):
    adj_np_array = np.array(adj.toarray(), dtype=np.float32)
    zero_mask = np.zeros_like(adj_np_array, dtype=np.float32)
    return zero_mask


def update_gradients_w(grads_vars, adj_all_mask):
    count = 0
    #print("update weight")
    for grad, var in grads_vars:
        if var.name == 'gcn/graphconvolution_1_vars/adj:0' or var.name == 'gcn/graphconvolution_2_vars/adj:0':
            adj_mask = tf.cast(tf.constant(adj_all_mask), tf.float32)
            grads_vars[count] = (tf.multiply(adj_mask, grad), var)
        count += 1
    return grads_vars


def update_gradients_adj(grads_vars, adj_p_mask):
    count = 0
    #print("update adj")
    temp_grad_adj1 = 0
    count1 = 0
    var1 = None
    count2 = 0
    var2 = None
    temp_grad_adj2 = 0
    for grad, var in grads_vars:
        if var.name == 'gcn/graphconvolution_1_adj_vars/adj:0':
            adj_mask = tf.cast(tf.constant(adj_p_mask), tf.float32)
            temp_grad_adj = tf.multiply(adj_mask, grad)
            transposed_temp_grad_adj = tf.transpose(temp_grad_adj)
            temp_grad_adj1 = tf.add(temp_grad_adj, transposed_temp_grad_adj)
            count1 = count
            var1 = var
        # if var.name == 'gcn/graphconvolution_2_vars/weights_0:0' or var.name == 'gcn/graphconvolution_1_vars/weights_0:0':
        #     weight_mask = tf.cast(tf.zeros_like(grad), tf.float32)
        #     grads_vars[count] = (tf.multiply(weight_mask, grad), var)
        if var.name == 'gcn/graphconvolution_2_adj_vars/adj:0':
            adj_mask = tf.cast(tf.constant(adj_p_mask), tf.float32)
            temp_grad_adj = tf.multiply(adj_mask, grad)
            transposed_temp_grad_adj = tf.transpose(temp_grad_adj)
            temp_grad_adj2 = tf.add(temp_grad_adj, transposed_temp_grad_adj)
            count2 = count
            var2 = var
        count += 1
    grad_adj = tf.divide(tf.add(temp_grad_adj1, temp_grad_adj2), 4)
    grads_vars[count1] = (grad_adj, var1)
    grads_vars[count2] = (grad_adj, var2)
    return grads_vars


def prune_adj1(adj, percent=10):
    pcen = np.percentile(abs(adj),percent)
    print ("percentile " + str(pcen))
    under_threshold = abs(adj)< pcen
    adj[under_threshold] = 0
    above_threshold = abs(adj)>= pcen
    #return [above_threshold,weight_arr]
    return adj


def prune_adj(oriadj, non_zero_idx, mask, percent):
    # instead of using all the weight values, including zeros to calculate percentitle, we use only nonzero
    adj = np.multiply(oriadj, mask)
    cur_non_zero_idx = (adj != 0)
    len_cur_non_zero_idx = len(cur_non_zero_idx)
    len_non_zero_idx = len(non_zero_idx)
    coverged = len_cur_non_zero_idx - len_non_zero_idx
    #print("coverged:", coverged)
    percent = (percent - coverged / len_non_zero_idx) * len_non_zero_idx / len_cur_non_zero_idx
    #print("percent:", percent)
    non_zero_adj = adj[adj != 0]
    pcen = np.percentile(abs(non_zero_adj), percent)
    #print ("percentile " + str(pcen))
    under_threshold = abs(adj) < pcen
    before = len(non_zero_adj)
    #print("before",before)
    adj[under_threshold] = 0
    non_zero_adj = adj[adj!=0]
    after = len(non_zero_adj)
    #print("after", after)
    above_threshold = abs(adj)>= pcen
    adj = np.add(adj, np.identity(adj.shape[0]))
    return adj


def prune_adj2(oriadj, non_zero_idx, percent):
    # instead of using all the weight values, including zeros to calculate percentitle, we use only nonzero
    original_prune_num = int((non_zero_idx / 2) * (percent/100))
    #print("orignal prune num", original_prune_num)
    adj = np.copy(oriadj)
    #print("non zero in adj: ", np.count_nonzero(adj))
    # cur_non_zero_idx = np.count_nonzero(adj)
    # len_cur_non_zero_idx = cur_non_zero_idx
    # len_non_zero_idx = non_zero_idx
    # coverged = len_cur_non_zero_idx - len_non_zero_idx
    #print("coverged:", coverged)
    #if coverged != 0:
    #    percent = int(((percent/100 - coverged / len_non_zero_idx) * len_non_zero_idx / len_cur_non_zero_idx)*100)
    print("percent:", percent)
    low_adj= np.tril(adj, -1)
    non_zero_low_adj = low_adj[low_adj != 0]
    
    #print("non zero is low adj: ", np.count_nonzero(non_zero_low_adj))
    #print("min non_zero_low_adj: ", min(abs(non_zero_low_adj)))
    #print("max: ", max(abs(non_zero_low_adj)))
    
    low_pcen = np.percentile(abs(non_zero_low_adj), percent)
    #print("percentile " + str(low_pcen))
    under_threshold = abs(low_adj) < low_pcen
    before = len(non_zero_low_adj)
    low_adj[under_threshold] = 0
    non_zero_low_adj = low_adj[low_adj != 0]
    after = len(non_zero_low_adj)

    rest_pruned = original_prune_num - (before - after)
    #print("rest prune", rest_pruned)
    if rest_pruned > 0:
        #print("non zero: ", np.count_nonzero(low_adj))
        mask_low_adj = (low_adj != 0)
        low_adj[low_adj == 0] = 2000000
        flat_indices = np.argpartition(low_adj.ravel(), rest_pruned - 1)[:rest_pruned]
        row_indices, col_indices = np.unravel_index(flat_indices, low_adj.shape)
        low_adj = np.multiply(low_adj, mask_low_adj)
        low_adj[row_indices, col_indices] = 0
        #print("after non zero: ", np.count_nonzero(low_adj))
        #print("non_zero in prune adj2: ", np.count_nonzero(low_adj))
    adj = low_adj + np.transpose(low_adj)
    adj = np.add(adj, np.identity(adj.shape[0]))
    return adj

def initialize(adj):
    res = np.zeros_like(adj)
    return res


def convertoadj(admm_adj):
    adj = np.copy(admm_adj)
    adj[adj != 0] = 1
    return adj


def testsymmetry(adj):
    res = np.subtract(adj, np.transpose(adj))
    return np.count_nonzero(res)


def isequal(adj1, adj2):
    a1 = np.array(adj1)
    a2 = np.array(adj2)
    return ((a1 == a2).all())


def zerolike(adj):
    return np.zeros_like(adj) + np.identity(adj.shape[0])


def initialize_uninitialized_global_variables(sess):
    """
    Only initializes the variables of a TensorFlow session that were not
    already initialized.
    :param sess: the TensorFlow session
    :return:
    """
    # List all global variables
    global_vars = tf.global_variables()

    # Find initialized status for all variables
    is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    is_initialized = sess.run(is_var_init)

    # List all variables that were not initialized previously
    not_initialized_vars = [var for (var, init) in
                            zip(global_vars, is_initialized) if not init]
    # Initialize all uninitialized variables found, if any
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))



def remove_file(path):
    pathTest = path + "_tmp"
    try:
        shutil.rmtree(pathTest)
    except OSError as e:
        print(e)
    else:
        print("The directory is deleted successfully")

