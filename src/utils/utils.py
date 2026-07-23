# -*- coding: UTF-8 -*-
import os
import torch
import datetime
import numpy as np
import random
from scipy.sparse import csr_matrix


LOSS_SMOOTHING_WINDOW = 5


def read_data_from_file(filename, flag=0):
    with open(filename, "r") as f: 
        lines = f.readlines()
        if flag == 0:
            data = [line.replace('\n','').split() for line in lines]
        elif flag == 1:
            data = [line.replace('\n','').split('::') for line in lines]
    return data

def read_data_from_file_int(filename, flag=0):
    with open(filename, "r") as f: 
        lines = f.readlines()
        if flag == 0:
            data = [str_list_to_int(line.split()) for line in lines]
        elif flag == 1:
            data = [str_list_to_int(line.split('::')) for line in lines]
    return data

def str_list_to_int(str_list):
    return [int(item) for item in str_list]

def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory - " + directory)

def get_user_dil_from_edgelist(edges):
    dil = {}
    for edge in edges:
        if dil.get(edge[0]) is None:
            dil[edge[0]] = []
        dil[edge[0]].append(edge[1])

    return dil

def get_user_item_set(edges):
    user_set = set()
    item_set = set()
    for edge in edges:
        user_set.add(edge[0])
        item_set.add(edge[1])
    return list(user_set), list(item_set)

def write_interactions_to_file(filename, data):

    with open(filename, 'w+') as f:
        for d in data:
            f.writelines('{}\t{}\n'.format(d[0],d[1]))

def batch_to_gpu(batch: dict, device) -> dict:
    if not isinstance(batch, dict):
        raise TypeError(
            "Expected DataLoader to return a batch dict. "
            "If fast_sampler is enabled, construct loaders with utils.build_data_loader()."
        )
    non_blocking = getattr(device, 'type', str(device)) == 'cuda'
    for c in batch:
        if type(batch[c]) is torch.Tensor:
            batch[c] = batch[c].to(device, non_blocking=non_blocking)
    return batch


def build_data_loader(dataset, batch_size, shuffle, num_workers, pin_memory,
                      persistent_workers, prefetch_factor):
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': bool(pin_memory),
    }
    use_fast_collate = dataset.args.fast_sampler and hasattr(dataset, 'collate_batch')
    if hasattr(dataset, '_use_fast_collate'):
        dataset._use_fast_collate = bool(use_fast_collate)
    if use_fast_collate:
        loader_kwargs['collate_fn'] = dataset.collate_batch
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = bool(persistent_workers)
        loader_kwargs['prefetch_factor'] = prefetch_factor
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


def squeeze_dict(batch: dict, dim=0) -> dict:
    for c in batch:
        if not torch.is_tensor(batch[c]):
            batch[c] = torch.from_numpy(batch[c])
        batch[c].squeeze_(dim)
    return batch

def check_dir(file_name: str):
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path):
        print('make dirs:', dir_path)
        os.makedirs(dir_path)

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_data_as_dict(corpus, data_type, data_idx):
    pth = os.path.join(corpus.snapshots_path, data_type+'_block'+str(data_idx))
    dat = np.array(read_data_from_file_int(pth))
    user_clicked_list = {}
    for user, item in dat:
        if user not in user_clicked_list:
            user_clicked_list[user] = []
        user_clicked_list[user].append(item)
    users, items = dat[:, 0], dat[:, 1]
    # user_item_csr = csr_matrix((np.ones(len(dat), dtype=np.float32), (users, items)),
    #                                     shape=(corpus.n_users, corpus.n_items))
    unique_items = np.array(list(set(items)))  
    return  user_clicked_list, users, unique_items
