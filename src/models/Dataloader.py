import torch
import logging
import os
import numpy as np
import copy
from random import randint
from torch.utils.data import Dataset as BaseDataset
from utils import utils
from scipy.sparse import csr_matrix
from time import time
import scipy.sparse as sp


class Dataset(BaseDataset):
    def __init__(self, args, corpus, data_type, data_idx):
        # self.model = model  # model object reference
        self.corpus = corpus  # reader object reference
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.time_idx = data_idx
        self.data_type = data_type

        self.train_file = os.path.join(corpus.snapshots_path, data_type+'_block'+str(data_idx))
        
        self.train_data = utils.read_data_from_file_int(self.train_file)
        self.train_data = np.array(self.train_data)

        print('train data shape',self.train_data.shape)


        self.trainUser = self.train_data[:, 0]
        self.trainItem = self.train_data[:, 1]
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser), dtype=np.float32), (self.trainUser, self.trainItem)),
									shape=(corpus.n_users, corpus.n_items))    
        self.Graph = None

        self.user_set = set(self.trainUser)
        self.item_set = set(self.trainItem)
        # self.num_users = len(self.user_set)
        # self.num_items = len(self.item_set)
        self.num_users = self.trainUser.max() + 1
        self.num_items = self.trainItem.max() + 1
        # get neighbor set for each user
        self.user_neigh = {}
        for user in self.user_set:
            self.user_neigh[user] = list(self.UserItemNet[user].indices)

        # for contrastive
        self.ItemUserNet = self.UserItemNet.T.tocsr()
        self.item_neigh = {item: list(self.ItemUserNet[item].indices) for item in self.item_set}

        # specially for neg sample
        hist_user_clicked_list, _, hist_unique_items = utils.load_data_as_dict(corpus, 'hist', data_idx)
        for user_id in hist_user_clicked_list:
            hist_user_clicked_list[user_id] = set(hist_user_clicked_list[user_id])
        self.hist_user_clicked_set = hist_user_clicked_list
        self.hist_unique_items = hist_unique_items

        
        # get neighbor set for each item
        # self.item_neigh = {}
        # for item in self.item_set:
        #     self.item_neigh[item] = list(self.UserItemNet[:, item].nonzero()[0])
        


    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, index: int) -> dict:
        #current = self._get_feed_dict(index)
        current = self._get_feed_dict_fast(index)
        return current

    def _get_feed_dict(self, index: int) -> dict:

        user_id, item_id = self.train_data[index]
        neg_items = self._sample_neg_items(user_id).squeeze()
        user_id, item_id = torch.tensor([user_id]), torch.tensor([item_id])
        item_id_ = torch.cat((item_id, neg_items), axis=-1)
        
        feed_dict = {'user_id': user_id, #(batch_size, )
                        'item_id': item_id_} #(batch_size, 1+neg_items)

        return feed_dict
    

    def _sample_neg_items(self, user_id):
        #num_neg = self.model.num_neg
        num_neg = self.args.num_neg


        neg_items = torch.zeros(size=(1, num_neg), dtype=torch.int64)
        #neg_items = torch.zeros(size=(num_neg), dtype=torch.int64)

        #for idx, user in enumerate(self.corpus.user_list[index:index_end]): # Automatic coverage?
        #for idx, user in enumerate(user_id): # Automatic coverage?

        user_clicked_set = copy.deepcopy(self.corpus.user_clicked_set[user_id])
        # By copying, it may not collide with other process with same user index
        for neg in range(num_neg):
            neg_item = self._randint_w_exclude(user_clicked_set)
            neg_items[0][neg] = neg_item
            # Skip below: one neg for train
            user_clicked_set = np.append(user_clicked_set, neg_item)

        return neg_items

    def _randint_w_exclude(self, clicked_set):
        randItem = randint(1, self.corpus.n_items-1)
        return self._randint_w_exclude(clicked_set) if randItem in clicked_set else randItem
    

    def _get_feed_dict_fast(self, index: int) -> dict:
        user_id, item_id = self.train_data[index]
        neg_items = self._sample_neg_items_fast(user_id)
        feed_dict = {
            'user_id': torch.tensor([user_id]),     # single
            'item_id': torch.cat([torch.tensor([item_id]), neg_items])
        }
        return feed_dict
    
    def _sample_neg_items_fast(self, user_id):
        num_neg = self.args.num_neg
        clicked_set = self.hist_user_clicked_set[user_id]
        sample_pool = self.hist_unique_items
        
        neg_items = []
        neg_set = set()
        while len(neg_items) < num_neg:
            samples = np.random.randint(0, len(sample_pool), size=num_neg * 2)
            for idx in samples:
                rand_item = sample_pool[idx]
                if rand_item not in clicked_set and rand_item not in neg_set:
                    neg_items.append(rand_item)
                    neg_set.add(rand_item)
                if len(neg_items) == num_neg:
                    break
        return torch.tensor(neg_items, dtype=torch.int64)
    

    # for GCN
    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.n_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self._device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            adj_mat_path = os.path.join(self.corpus.snapshots_path, self.data_type+'_adj_mat_t{}.npz'.format(self.time_idx))
            try:
                pre_adj_mat = sp.load_npz(adj_mat_path)
                print("successfully loaded adjacency matrix...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                # norm_adj = self.get_norm_adj()
                norm_adj = self.get_norm_adj_optimized()
                end = time()
                print(f"computing time {end-s}s, save norm_mat..")
                sp.save_npz(adj_mat_path, norm_adj)

            # if self.split == True:
            #     self.Graph = self._split_A_hat(norm_adj)
            #     print("done split matrix")
            # else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self._device)
                #print("don't split the matrix")
        return self.Graph
    
    def get_norm_adj(self):
        adj_mat = sp.dok_matrix((self.corpus.n_users + self.corpus.n_items, self.corpus.n_users + self.corpus.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.UserItemNet.tolil()
        adj_mat[:self.corpus.n_users, self.corpus.n_users:] = R
        adj_mat[self.corpus.n_users:, :self.corpus.n_users] = R.T
        adj_mat = adj_mat.todok()
        # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
        
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        return norm_adj

    def get_norm_adj_optimized(self):
        """
        Optimized generation of the normalized adjacency matrix for GNN-based recommendation.
        Uses direct COO coordinate construction to build the bipartite graph structure:
        Matrix A = [[0, R], [R.T, 0]]
        """
        n_users = self.corpus.n_users
        n_items = self.corpus.n_items
        
        # 1. Directly obtain COO coordinates from the User-Item interaction matrix (R)
        # R shape is assumed to be (n_users, n_items)
        R_coo = self.UserItemNet.tocoo()
        
        # 2. Construct coordinates for the symmetric bipartite graph
        # Top-right block (0, R): row indices remain same, column indices shift by n_users
        row_upper = R_coo.row
        col_upper = R_coo.col + n_users
        
        # Bottom-left block (R.T, 0): row indices shift by n_users, column indices from R's original rows
        row_lower = R_coo.col + n_users
        col_lower = R_coo.row
        
        # Concatenate all coordinates and data to form the full adjacency information
        rows = np.concatenate([row_upper, row_lower])
        cols = np.concatenate([col_upper, col_lower])
        data = np.concatenate([R_coo.data, R_coo.data])
        
        # 3. Construct the large adjacency matrix using COO format for efficiency
        adj_mat = sp.coo_matrix((data, (rows, cols)), 
                                shape=(n_users + n_items, n_users + n_items))
        
        # 4. Normalization process: Symmetric normalization D^-0.5 * A * D^-0.5
        # Convert to CSR format for significantly faster row-sum and matrix operations
        adj_mat = adj_mat.tocsr() 
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        
        # Calculate the inverse square root of degrees, handling division by zero for isolated nodes
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        
        """ 
        # inefficient, memory-consuming
        # Compute the normalized matrix: norm_adj = D^-0.5 * A * D^-0.5
        d_mat = sp.diags(d_inv_sqrt)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        return norm_adj.tocsr() 
        """

        A = adj_mat
        data, indices, indptr = A.data, A.indices, A.indptr
        # Normilze：A[i, j] = d_inv_sqrt[i] * A[i, j] * d_inv_sqrt[j]
        for i in range(A.shape[0]):
            start, end = indptr[i], indptr[i+1]
            data[start:end] = data[start:end] * d_inv_sqrt[i] * d_inv_sqrt[indices[start:end]]
        return adj_mat