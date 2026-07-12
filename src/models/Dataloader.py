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
        self._use_fast_collate = False
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
        hist_user_clicked_list, _, _ = utils.load_data_as_dict(corpus, 'hist', data_idx)
        for user_id in hist_user_clicked_list:
            hist_user_clicked_list[user_id] = set(hist_user_clicked_list[user_id])
        self.hist_user_clicked_set = hist_user_clicked_list
        self.hist_pair_codes = self._build_hist_pair_codes(hist_user_clicked_list)
        self.current_unique_items = np.unique(self.trainItem)
        self.current_unique_item_set = set(int(item) for item in self.current_unique_items)
        self.all_items = np.arange(corpus.n_items, dtype=np.int64)

        
        # get neighbor set for each item
        # self.item_neigh = {}
        # for item in self.item_set:
        #     self.item_neigh[item] = list(self.UserItemNet[:, item].nonzero()[0])
        


    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, index: int) -> dict:
        if getattr(self.args, 'fast_sampler', 1) and self._use_fast_collate:
            return index
        #current = self._get_feed_dict(index)
        current = self._get_feed_dict_fast(index)
        return current

    def collate_batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        batch_data = self.train_data[indices]
        user_ids = batch_data[:, 0].astype(np.int64, copy=False)
        pos_items = batch_data[:, 1].astype(np.int64, copy=False)
        if self._use_legacy_aux_neg_sampling():
            neg_items = np.stack(
                [self._sample_neg_items_fast(int(user_id)).numpy() for user_id in user_ids],
                axis=0,
            )
        else:
            neg_items = self._sample_neg_items_batch(user_ids)

        item_ids = np.empty((len(indices), self.args.num_neg + 1), dtype=np.int64)
        item_ids[:, 0] = pos_items
        item_ids[:, 1:] = neg_items

        return {
            'user_id': torch.from_numpy(user_ids.reshape(-1, 1)),
            'item_id': torch.from_numpy(item_ids),
        }

    def _use_legacy_aux_neg_sampling(self):
        legacy_flag = getattr(self.args, 'legacy_aux_neg_sampler', -1)
        legacy_models = {'PISA_LGN', 'Contrastive_LGN'}
        if legacy_flag < 0:
            return getattr(self.args, 'model_name', '') in legacy_models
        return bool(legacy_flag) and getattr(self.args, 'model_name', '') in legacy_models

    def _build_hist_pair_codes(self, hist_user_clicked_list):
        user_chunks, item_chunks = [], []
        for user_id, item_set in hist_user_clicked_list.items():
            if len(item_set) == 0:
                continue
            items = np.fromiter(item_set, dtype=np.int64)
            user_chunks.append(np.full(len(items), int(user_id), dtype=np.int64))
            item_chunks.append(items)

        if not user_chunks:
            return np.empty(0, dtype=np.int64)

        users = np.concatenate(user_chunks)
        items = np.concatenate(item_chunks)
        return np.sort(users * np.int64(self.corpus.n_items) + items)

    def _hist_clicked_mask_batch(self, user_ids, candidate_items):
        if self.hist_pair_codes.size == 0:
            return np.zeros(candidate_items.shape, dtype=bool)

        codes = user_ids.astype(np.int64)[:, None] * np.int64(self.corpus.n_items)
        codes = (codes + candidate_items.astype(np.int64)).reshape(-1)
        positions = np.searchsorted(self.hist_pair_codes, codes)
        in_bounds = positions < self.hist_pair_codes.size

        clicked = np.zeros(codes.shape, dtype=bool)
        clicked[in_bounds] = self.hist_pair_codes[positions[in_bounds]] == codes[in_bounds]
        return clicked.reshape(candidate_items.shape)

    @staticmethod
    def _row_duplicate_mask(values):
        if values.shape[1] <= 1:
            return np.zeros(values.shape, dtype=bool)

        order = np.argsort(values, axis=1, kind='mergesort')
        sorted_values = np.take_along_axis(values, order, axis=1)
        duplicate_sorted = np.zeros(values.shape, dtype=bool)
        duplicate_sorted[:, 1:] = sorted_values[:, 1:] == sorted_values[:, :-1]

        duplicate = np.zeros(values.shape, dtype=bool)
        np.put_along_axis(duplicate, order, duplicate_sorted, axis=1)
        return duplicate

    def _sample_neg_items_batch(self, user_ids):
        num_neg = self.args.num_neg
        pool = self.current_unique_items
        if len(pool) == 0:
            raise ValueError('Cannot sample negatives from an empty item pool')

        batch_size = len(user_ids)
        neg_items = np.full((batch_size, num_neg), -1, dtype=np.int64)
        selected_counts = np.zeros(batch_size, dtype=np.int64)
        candidate_width = max(num_neg * 16, 64)

        for _ in range(3):
            active_rows = np.flatnonzero(selected_counts < num_neg)
            if len(active_rows) == 0:
                break

            active_users = user_ids[active_rows]
            candidate_idx = np.random.randint(0, len(pool), size=(len(active_rows), candidate_width))
            candidates = pool[candidate_idx]

            valid = ~self._hist_clicked_mask_batch(active_users, candidates)
            valid &= ~self._row_duplicate_mask(candidates)

            existing = neg_items[active_rows]
            valid &= ~(candidates[:, :, None] == existing[:, None, :]).any(axis=2)

            ranks = np.cumsum(valid, axis=1) + selected_counts[active_rows, None]
            take = valid & (ranks <= num_neg)
            local_rows, cols = np.nonzero(take)
            if len(local_rows) == 0:
                continue

            target_rows = active_rows[local_rows]
            target_cols = ranks[local_rows, cols] - 1
            neg_items[target_rows, target_cols] = candidates[local_rows, cols]
            selected_counts += np.bincount(target_rows, minlength=batch_size)

        for row in np.flatnonzero(selected_counts < num_neg):
            user_id = int(user_ids[row])
            clicked_set = self.hist_user_clicked_set.get(user_id, set())
            valid_pool = np.array(
                [item for item in pool if int(item) not in clicked_set],
                dtype=np.int64,
            )
            if len(valid_pool) == 0:
                raise ValueError(f'No available negative items for user {user_id}')

            if len(valid_pool) < num_neg:
                repeats = num_neg // len(valid_pool)
                remainder = num_neg % len(valid_pool)
                fill = np.tile(valid_pool, repeats)
                neg_items[row, :len(fill)] = fill
                if remainder:
                    neg_items[row, repeats * len(valid_pool):] = np.random.choice(
                        valid_pool,
                        size=remainder,
                        replace=False,
                    )
                continue

            chosen = set(int(item) for item in neg_items[row, :selected_counts[row]])
            while selected_counts[row] < num_neg:
                item = int(valid_pool[np.random.randint(0, len(valid_pool))])
                if item in chosen:
                    continue
                neg_items[row, selected_counts[row]] = item
                chosen.add(item)
                selected_counts[row] += 1

        return neg_items

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
        clicked_set = self.hist_user_clicked_set.get(user_id, set())
        sample_pool = self.current_unique_items

        if len(sample_pool) - len(clicked_set) < num_neg:
            # Not enough distinct negatives: reuse each valid item evenly
            targets = np.array(
                [item for item in sample_pool if item not in clicked_set],
                dtype=np.int64,
            )
            repeats = num_neg // len(targets)
            remainder = num_neg % len(targets)
            neg_items = list(np.tile(targets, repeats))
            neg_items.extend(np.random.choice(targets, size=remainder, replace=False))
            return torch.tensor(neg_items, dtype=torch.int64)

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
