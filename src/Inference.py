import torch
import numpy as np
import os
from utils import utils
import math
import logging
from time import perf_counter


def _sync_if_cuda(device):
    if device is not None and torch.cuda.is_available():
        device = torch.device(device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = []
    #归一化折扣累计增益
    NDCG = []
    #平均倒数排名
    MRR = []
    #user_list = [1309,1662,1961] # stable
    #user_list = [1941,1909,1799,1662,1626,1624,1600] # unstable
    for index in range(len(topN)):
        # print(f'top {topN[index]}\n')
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        cnt = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                #是否计算mrr
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                #理想的折扣累计增益
                idcg = 0
                #剩余的理想命中计数
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
                cnt += 1
            # else: 
            #     print('OPS')
#             if i in user_list:
#                 print(f'user {i}')
# #                 print(predictedIndices[i])
# #                 print(GroundTruth[i])
#                 print(userHit / len(GroundTruth[i]))
#                 print(ndcg)
        precision.append(round(sumForPrecision / cnt, 4))
        recall.append(round(sumForRecall / cnt, 4))
        NDCG.append(round(sumForNdcg / cnt, 4))
        MRR.append(round(sumForMRR / cnt, 4))
        
    return recall, NDCG, MRR, precision


def computeTopNAccuracy_fast(GroundTruth, predictedIndices, topN, lite = False):
    """
    vector-based metric cal
    GroundTruth: List of sets ( set = clicked items)
    predictedIndices: list of NumPy array [max_K] or [num_users, max_K]
    topN: List of int ( [10, 20, 50, 100])
    """
    # 1. to numpy array
    pred = np.vstack(predictedIndices)
    num_users = len(pred)
    max_k = pred.shape[1]

    # 2. Hit Matrix
    hits = np.zeros(pred.shape, dtype=np.int8)
    actual_counts = np.fromiter((len(gt) for gt in GroundTruth), dtype=np.int32, count=len(GroundTruth))

    for i in range(num_users):
        gt_set = GroundTruth[i]
        hits[i, :] = [1 if item in gt_set else 0 for item in pred[i]]

    # 3. cal stat
    # each row are cum hitnum of 1~max_k positions
    hit_cumsum = np.cumsum(hits, axis=1) 

    results = {'recall': [], 'ndcg': [], 'mrr': [], 'precision': []}

    if lite:
        for k in topN:
            k_idx = k - 1
            user_recalls = hit_cumsum[:, k_idx] / actual_counts
            results['recall'].append(round(np.mean(user_recalls), 4))
        return results['recall'], None, None, None
    
    # DCG: score / log2(rank + 1) ->  rank is j+1 so that log2(j+2)
    weights = 1.0 / np.log2(np.arange(2, max_k + 2))
    dcg_matrix = np.cumsum(hits * weights, axis=1)
    weights_cumsum = np.cumsum(weights)

    for k in topN:
        k_idx = k - 1
        
        # --- Recall & Precision ---
        user_recalls = hit_cumsum[:, k_idx] / actual_counts
        user_precisions = hit_cumsum[:, k_idx] / k
        
        # --- MRR ---
        # Find the index of the first hit (value of 1) in each row
        # argmax returns the index of the first occurrence of the maximum value
        first_hit_idx = np.argmax(hits[:, :k], axis=1)
        #Caution: Since argmax returns index 0 when a row contains only zeros, ensure has_hit is used for validation.
        has_hit = np.max(hits[:, :k], axis=1) > 0
        user_mrrs = np.where(has_hit, 1.0 / (first_hit_idx + 1), 0.0)

        # --- NDCG ---
        current_dcg = dcg_matrix[:, k_idx]
        # IDCG sum of min(k, len(gt)) elements
        idcg_counts = np.minimum(k, actual_counts)
        user_idcgs = weights_cumsum[idcg_counts - 1]
        user_ndcgs = current_dcg / user_idcgs

        # avg over user
        results['recall'].append(round(np.mean(user_recalls), 4))
        results['precision'].append(round(np.mean(user_precisions), 4))
        results['mrr'].append(round(np.mean(user_mrrs), 4))
        results['ndcg'].append(round(np.mean(user_ndcgs), 4))

    return results['recall'], results['ndcg'], results['mrr'], results['precision']


def Test_group(args, model, corpus, data_type, data_idx, group_files):
    """
    Args:
        group_files: Dictionary containing paths to user group files
            e.g., {"dynamic": "dynamic_users.txt", "static": "static_users.txt", "intermediate": "intermediate_users.txt"}
    """

    batch_size = args.eval_batch_size if getattr(args, 'eval_batch_size', 0) > 0 else args.batch_size
    model.eval()

    # Load user groups from files
    user_groups = {}
    for group_name, file_path in group_files.items():
        with open(file_path, "r") as f:
            user_groups[group_name] = set(int(line.strip()) for line in f)

    test_file = os.path.join(corpus.snapshots_path, data_type + '_block' + str(data_idx))
    test_data = utils.read_data_from_file_int(test_file)
    test_data = np.array(test_data)
    test_pos_items = {}
    for user, item in test_data:
        if user not in test_pos_items:
            test_pos_items[user] = []
        test_pos_items[user].append(item)

    hist_file = os.path.join(corpus.snapshots_path, 'hist_block' + str(data_idx))
    hist_data = utils.read_data_from_file_int(hist_file)
    hist_data = np.array(hist_data)
    hist_pos_items = {}
    for user, item in hist_data:
        if user not in hist_pos_items:
            hist_pos_items[user] = []
        hist_pos_items[user].append(item)

    Ks = [10, 20, 50, 100]
    max_K = max(Ks)

    group_results = {}

    with torch.no_grad():
        # Evaluate each user group
        for group_name, group_users in user_groups.items():
            users = [u for u in test_pos_items.keys() if u in group_users]
            if not users:
                continue

            users_list = []
            rating_list = []
            ground_truth_list = []

            n_batch = len(users) // batch_size
            if len(users) % batch_size != 0:
                n_batch += 1

            for i in range(n_batch):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(users))
                batch_users = users[start:end]

                all_pos = []
                for user in batch_users:
                    # Cold-start users
                    if user not in hist_pos_items:
                        all_pos.append([])
                    else:
                        all_pos.append(hist_pos_items[user])

                ground_truth = []
                for user in batch_users:
                    ground_truth.append(test_pos_items[user])

                user_id = torch.tensor(batch_users, dtype=torch.int64).to(model._device)
                item_id = torch.tensor(np.arange(corpus.n_items), dtype=torch.int64).to(model._device)

                scores = model.infer_user_scores(user_id, item_id)

                exclude_index = []
                exclude_items = []
                for i, items in enumerate(all_pos):
                    exclude_index.extend([i] * len(items))
                    exclude_items.extend(items)
                if exclude_index:
                    scores[
                        torch.tensor(exclude_index, dtype=torch.long, device=model._device),
                        torch.tensor(exclude_items, dtype=torch.long, device=model._device),
                    ] = -torch.inf

                _, rating_K = torch.topk(scores, k=max_K, dim=1)

                users_list.append(batch_users)
                rating_list.extend(rating_K.cpu())
                ground_truth_list.extend(ground_truth)

            recall, NDCG, MRR, precision = computeTopNAccuracy(ground_truth_list, rating_list, Ks)
            group_results[group_name] = (recall, NDCG, MRR, precision)

    return group_results


# inference incl. cold user & cold item, their embedding are random.
def Test(args, model, corpus, data_type, data_idx):
    batch_size = args.eval_batch_size if getattr(args, 'eval_batch_size', 0) > 0 else args.batch_size
    model.eval()

    #data_type = 'val'

    #dataset = Dataloader.Dataset(model, args, corpus, data_type, idx)
    test_file = os.path.join(corpus.snapshots_path, data_type+'_block'+str(data_idx))
    test_data = utils.read_data_from_file_int(test_file)
    test_data = np.array(test_data)
    print(f'{data_type} data shape', test_data.shape)
    test_data = test_data
    test_pos_items = {}
    for user, item in test_data:
        if user not in test_pos_items:
            test_pos_items[user] = []
        test_pos_items[user].append(item)

    hist_file = os.path.join(corpus.snapshots_path, 'hist_block'+str(data_idx))
    hist_data = utils.read_data_from_file_int(hist_file)
    
    hist_data = np.array(hist_data)
    hist_pos_items = {}
    for user, item in hist_data:
        if user not in hist_pos_items:
            hist_pos_items[user] = []
        hist_pos_items[user].append(item)

    Ks = [10,20,50,100]
    max_K = max(Ks)

    users_list = []
    rating_list = []
    ground_truth_list = []

    with torch.no_grad():
        users = list(test_pos_items.keys())
        n_batch = len(users) // batch_size
        if len(users) % batch_size != 0:
            n_batch += 1
        
        for i in range(n_batch):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(users))
            batch_users = users[start:end]

            all_pos = []
            for user in batch_users:
                # cold-start users
                if user not in hist_pos_items:
                    all_pos.append([])
                else:
                    all_pos.append(hist_pos_items[user])
            
            ground_truth = []
            for user in batch_users:
                ground_truth.append(test_pos_items[user])

            user_id = torch.tensor(batch_users, dtype=torch.int64).to(model._device)
            item_id = torch.tensor(np.arange(corpus.n_items), dtype=torch.int64).to(model._device)

            scores = model.infer_user_scores(user_id, item_id)

            exclude_index = []
            exclude_items = []
            for i, items in enumerate(all_pos):
                exclude_index.extend([i] * len(items))
                exclude_items.extend(items)
            if exclude_index:
                scores[
                    torch.tensor(exclude_index, dtype=torch.long, device=model._device),
                    torch.tensor(exclude_items, dtype=torch.long, device=model._device),
                ] = -torch.inf

            _, rating_K = torch.topk(scores, k=max_K, dim=1)
            
            users_list.append(batch_users)
            rating_list.extend(rating_K.cpu())
            ground_truth_list.extend(ground_truth)
        
        assert n_batch == len(users_list)
        
        recall, NDCG, MRR, precision = computeTopNAccuracy(ground_truth_list, rating_list, Ks)

        # results = [recall, NDCG, MRR, precision]
        # string_results = {f'{metric}@{K}': v for metric, v in zip(['Recall', 'NDCG', 'MRR', 'Precision'], results) for K, v in zip(Ks, v)}

        return recall, NDCG, MRR, precision


# inference excl. cold user & cold item
def Test_excl_cold(args, model, test_loads, hist_loads, lite = False):
    if getattr(args, 'compare_vectorized_eval', 0) and not getattr(args, '_eval_compare_active', False):
        return Test_excl_cold_selected(args, model, test_loads, hist_loads, lite=lite, label='direct')

    batch_size = args.eval_batch_size if getattr(args, 'eval_batch_size', 0) > 0 else args.batch_size
    model.eval()
    device = model._device

    test_user_clicked_list, testUsers, _ = test_loads
    hist_user_clicked_list, hist_users_list, hist_unique_items = hist_loads
    
    with torch.no_grad():

        # --- GCN propgation ---
        all_users_emb, all_items_emb = model.computer() 
        
        # target Item Embedding (excl. cold Item)
        target_items_tensor = torch.from_numpy(hist_unique_items).to(device)
        target_items_emb = all_items_emb[target_items_tensor] 
        

        # target user
        hist_users_set = set(hist_users_list)
        valid_test_users = [u for u in testUsers if u in hist_users_set]
        valid_test_user_clicked_set = {u:set(test_user_clicked_list[u]) for u in valid_test_users}
        
        # --- proprecess Mask looktable ---
        max_item_id = hist_unique_items.max()
        item_mapping = torch.full((max_item_id + 1,), -1, dtype=torch.long, device=device)
        item_mapping[torch.from_numpy(hist_unique_items).to(device)] = torch.arange(len(hist_unique_items), device=device)
        
        Ks = [10, 20, 50, 100]
        max_K = max(Ks)

        rating_list = []
        ground_truth_list = []
         
        for i in range(0, len(valid_test_users), batch_size):
            batch_users = valid_test_users[i : i + batch_size]
            
            # --- read user embedding w/o GCN propgation ---
            user_idx = torch.tensor(batch_users, dtype=torch.long, device=device)
            current_user_emb = all_users_emb[user_idx] # [batch_size, dim]
            
            scores = torch.matmul(current_user_emb, target_items_emb.t()) # [batch_size, num_items]

            # ---  Mask clicked items---
            for idx, u in enumerate(batch_users):
                clicked_items = torch.tensor(hist_user_clicked_list[u], dtype=torch.long, device=device)
                mapped_indices = item_mapping[clicked_items]
                valid_indices = mapped_indices[mapped_indices != -1]
                if len(valid_indices) > 0:
                    scores[idx, valid_indices] = -1e9

            # --- GPU Top-K ---
            #_, rating_K = torch.topk(scores, k=max_K, dim=1)
            _, col_indices = torch.topk(scores, k=max_K, dim=1)
            rating_K = target_items_tensor[col_indices]
            
            rating_list.extend(rating_K.cpu().numpy())
            ground_truth_list.extend([valid_test_user_clicked_set[u] for u in batch_users])

        recall, NDCG, MRR, precision = computeTopNAccuracy_fast(ground_truth_list, rating_list, Ks, lite)

        return recall, NDCG, MRR, precision


def _pairs_overflow(max_left, right_size, max_right):
    int64_max = np.iinfo(np.int64).max
    return max_left > (int64_max - max_right) // right_size


def _flatten_user_item_pairs(user_items, users, item_mapping, device):
    rows = []
    cols = []
    for row, user in enumerate(users):
        items = user_items.get(user, [])
        if len(items) == 0:
            continue
        mapped = item_mapping[torch.as_tensor(items, dtype=torch.long, device=device)]
        mapped = mapped[mapped != -1]
        if len(mapped) == 0:
            continue
        rows.append(torch.full((len(mapped),), row, dtype=torch.long, device=device))
        cols.append(mapped)

    if not rows:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    return torch.cat(rows), torch.cat(cols)


def Test_excl_cold_vectorized(args, model, test_loads, hist_loads, lite=False):
    batch_size = args.eval_batch_size if getattr(args, 'eval_batch_size', 0) > 0 else args.batch_size
    model.eval()
    device = model._device

    test_user_clicked_list, testUsers, _ = test_loads
    hist_user_clicked_list, hist_users_list, hist_unique_items = hist_loads

    with torch.no_grad():
        all_users_emb, all_items_emb = model.computer()

        target_items_tensor = torch.from_numpy(hist_unique_items).to(device)
        target_items_emb = all_items_emb[target_items_tensor]

        hist_users_set = set(hist_users_list)
        valid_test_users = [u for u in testUsers if u in hist_users_set]
        if len(valid_test_users) == 0:
            return [], None, None, None

        Ks = [10, 20, 50, 100]
        max_K = max(Ks)

        max_item_id = int(hist_unique_items.max())
        item_mapping = torch.full((max_item_id + 1,), -1, dtype=torch.long, device=device)
        item_mapping[target_items_tensor] = torch.arange(len(hist_unique_items), device=device)

        valid_test_user_clicked_set = {u: set(test_user_clicked_list[u]) for u in valid_test_users}
        test_item_counts = np.array(
            [len(valid_test_user_clicked_set[u]) for u in valid_test_users],
            dtype=np.int64,
        )
        hist_mask_rows, hist_mask_cols = _flatten_user_item_pairs(
            hist_user_clicked_list,
            valid_test_users,
            item_mapping,
            device,
        )

        overflow = _pairs_overflow(
            max_left=max(len(valid_test_users) - 1, 0),
            right_size=model.item_num,
            max_right=model.item_num - 1,
        )
        if overflow:
            logging.warning('Vectorized eval pair encoding would overflow int64; falling back to Test_excl_cold')
            return Test_excl_cold(args, model, test_loads, hist_loads, lite=lite)

        truth_codes = []
        for row, user in enumerate(valid_test_users):
            items = np.asarray(list(valid_test_user_clicked_set[user]), dtype=np.int64)
            if len(items) == 0:
                continue
            truth_codes.append(row * np.int64(model.item_num) + items)
        truth_codes = np.sort(np.concatenate(truth_codes)) if truth_codes else np.empty(0, dtype=np.int64)

        rating_rows = []
        rating_items = []
        for start in range(0, len(valid_test_users), batch_size):
            end = min(start + batch_size, len(valid_test_users))
            batch_users = valid_test_users[start:end]
            user_idx = torch.tensor(batch_users, dtype=torch.long, device=device)
            scores = torch.matmul(all_users_emb[user_idx], target_items_emb.t())

            mask = (hist_mask_rows >= start) & (hist_mask_rows < end)
            if mask.any():
                scores[hist_mask_rows[mask] - start, hist_mask_cols[mask]] = -1e9

            _, col_indices = torch.topk(scores, k=max_K, dim=1)
            rating_items.append(target_items_tensor[col_indices].cpu().numpy())
            row_ids = np.arange(start, end, dtype=np.int64)[:, None]
            rating_rows.append(np.broadcast_to(row_ids, (end - start, max_K)).copy())

        pred_items = np.vstack(rating_items)
        pred_rows = np.vstack(rating_rows)
        pred_codes = pred_rows * np.int64(model.item_num) + pred_items.astype(np.int64)
        positions = np.searchsorted(truth_codes, pred_codes.reshape(-1))
        in_bounds = positions < len(truth_codes)
        hits_flat = np.zeros(pred_codes.size, dtype=np.int8)
        hits_flat[in_bounds] = truth_codes[positions[in_bounds]] == pred_codes.reshape(-1)[in_bounds]
        hits = hits_flat.reshape(pred_codes.shape)

        hit_cumsum = np.cumsum(hits, axis=1)
        results = {'recall': [], 'ndcg': [], 'mrr': [], 'precision': []}

        if lite:
            for k in Ks:
                recalls = hit_cumsum[:, k - 1] / test_item_counts
                results['recall'].append(round(float(np.mean(recalls)), 4))
            return results['recall'], None, None, None

        weights = 1.0 / np.log2(np.arange(2, max_K + 2))
        dcg_matrix = np.cumsum(hits * weights, axis=1)
        weights_cumsum = np.cumsum(weights)

        for k in Ks:
            k_idx = k - 1
            recalls = hit_cumsum[:, k_idx] / test_item_counts
            precisions = hit_cumsum[:, k_idx] / k

            first_hit_idx = np.argmax(hits[:, :k], axis=1)
            has_hit = np.max(hits[:, :k], axis=1) > 0
            mrrs = np.where(has_hit, 1.0 / (first_hit_idx + 1), 0.0)

            idcg_counts = np.minimum(k, test_item_counts)
            idcgs = weights_cumsum[idcg_counts - 1]
            ndcgs = dcg_matrix[:, k_idx] / idcgs

            results['recall'].append(round(float(np.mean(recalls)), 4))
            results['ndcg'].append(round(float(np.mean(ndcgs)), 4))
            results['mrr'].append(round(float(np.mean(mrrs)), 4))
            results['precision'].append(round(float(np.mean(precisions)), 4))

        return results['recall'], results['ndcg'], results['mrr'], results['precision']


def Test_excl_cold_selected(args, model, test_loads, hist_loads, lite=False, label=''):
    if getattr(args, 'compare_vectorized_eval', 0):
        start_msg = f'[EvalCompare:{label}] start lite={lite}'
        print(start_msg, flush=True)
        logging.info(start_msg)
        had_compare_active = hasattr(args, '_eval_compare_active')
        previous_compare_active = getattr(args, '_eval_compare_active', False)
        args._eval_compare_active = True
        order_num = int(np.random.randint(0, 2))
        order = 'old_first' if order_num % 2 else 'vectorized_first'
        try:
            if order_num % 2:
                _sync_if_cuda(getattr(model, '_device', None))
                old_start = perf_counter()
                old_results = Test_excl_cold(args, model, test_loads, hist_loads, lite=lite)
                _sync_if_cuda(getattr(model, '_device', None))
                old_time = perf_counter() - old_start

                _sync_if_cuda(getattr(model, '_device', None))
                vec_start = perf_counter()
                vec_results = Test_excl_cold_vectorized(args, model, test_loads, hist_loads, lite=lite)
                _sync_if_cuda(getattr(model, '_device', None))
                vec_time = perf_counter() - vec_start
            else:
                _sync_if_cuda(getattr(model, '_device', None))
                vec_start = perf_counter()
                vec_results = Test_excl_cold_vectorized(args, model, test_loads, hist_loads, lite=lite)
                _sync_if_cuda(getattr(model, '_device', None))
                vec_time = perf_counter() - vec_start

                _sync_if_cuda(getattr(model, '_device', None))
                old_start = perf_counter()
                old_results = Test_excl_cold(args, model, test_loads, hist_loads, lite=lite)
                _sync_if_cuda(getattr(model, '_device', None))
                old_time = perf_counter() - old_start
        finally:
            if had_compare_active:
                args._eval_compare_active = previous_compare_active
            else:
                delattr(args, '_eval_compare_active')
        diffs = []
        for old_metric, vec_metric in zip(old_results, vec_results):
            if old_metric is None or vec_metric is None:
                continue
            diffs.extend(np.abs(np.asarray(old_metric) - np.asarray(vec_metric)).tolist())
        max_diff = max(diffs) if diffs else 0.0
        test_user_clicked_list, testUsers, _ = test_loads
        _, hist_users_list, hist_unique_items = hist_loads
        hist_users_set = set(hist_users_list)
        valid_users = [u for u in testUsers if u in hist_users_set]
        gt_pairs = sum(len(set(test_user_clicked_list[u])) for u in valid_users)
        speedup = old_time / vec_time if vec_time > 0 else float('inf')
        msg = (
            f'[EvalCompare:{label}] lite={lite} valid_users={len(valid_users)} '
            f'target_items={len(hist_unique_items)} gt_pairs={gt_pairs} '
            f'order_num={order_num} order={order} '
            f'old_time={old_time:.4f}s vectorized_time={vec_time:.4f}s '
            f'speedup={speedup:.2f}x '
            f'max_abs_diff={max_diff:.8f} old={old_results} vectorized={vec_results}'
        )
        print(msg, flush=True)
        logging.info(msg)
        return old_results

    if getattr(args, 'vectorized_eval', 0):
        return Test_excl_cold_vectorized(args, model, test_loads, hist_loads, lite=lite)

    return Test_excl_cold(args, model, test_loads, hist_loads, lite=lite)


def print_results(loss, valid_result, test_result):
    result_str = ''
    """output the evaluation results."""
    if loss is not None:
        logging.info("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        logging.info("[Valid]: Recall: {} NDCG: {} MRR: {} Precision: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
                # result_str += 'Top-10\n'
        result_str += 'Recall@10\t' + str(valid_result[0][0]) + '\n'
        result_str += 'NDCG@10\t' + str(valid_result[1][0]) + '\n'
        # result_str += 'Top-20\n'
        result_str += 'Recall@20\t' + str(valid_result[0][1]) + '\n'
        result_str += 'NDCG@20\t' + str(valid_result[1][1]) + '\n'
        # result_str += 'Top-50\n'
        result_str += 'Recall@50\t' + str(valid_result[0][2]) + '\n'
        result_str += 'NDCG@50\t' + str(valid_result[1][2]) + '\n'
    if test_result is not None: 
        logging.info("[Test]: Recall: {} NDCG: {} MRR: {} Precision: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))
        
        # result_str += 'Top-10\n'
        result_str += 'Recall@10\t' + str(test_result[0][0]) + '\n'
        result_str += 'NDCG@10\t' + str(test_result[1][0]) + '\n'
        # result_str += 'Top-20\n'
        result_str += 'Recall@20\t' + str(test_result[0][1]) + '\n'
        result_str += 'NDCG@20\t' + str(test_result[1][1]) + '\n'
        # result_str += 'Top-50\n'
        result_str += 'Recall@50\t' + str(test_result[0][2]) + '\n'
        result_str += 'NDCG@50\t' + str(test_result[1][2]) + '\n'

    return result_str
