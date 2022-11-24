from cmath import log
import enum
import os 
import torch
import random
import numpy as np
from typing import Optional, Tuple
import torch
from torch import Tensor
from typing import Optional, Tuple
from sklearn.metrics import precision_score,recall_score,ndcg_score
truncate_len = 30
import pandas as pd

def topk_recall_precision_ndgc(users, items, targets, recall_score, user_item_rating_df, k):
    precision,recall,ndcg = [],[],[]

    recall_score = recall_score.topk(k)[1].numpy()
    
    for uid, iid, label, recall_item in zip(users, items, targets, recall_score):
        precision.append(np.isin(label, recall_item))
        item_rating = user_item_rating_df[user_item_rating_df["userid"] ==  uid]
        item_rating_dic = dict(zip(item_rating["itemid"], user_item_rating_df["rate"]))
        recall_true = [item_rating_dic[item]  if item in item_rating_dic else 0 for item in recall_item]

        if len(np.where(recall_true == iid)[0]) == 0:
            recall.append(0)
        else:
            recall.append(1 / len(recall_true))

        dcg = recall_true / np.log2(np.arange(1, k + 1) + 1)
        dcg = np.sum(dcg)

        best_recall_true = sorted(list(item_rating_dic.values()))
        if len(best_recall_true) < k:
            best_recall_true += [0] * (k - len(best_recall_true))
        best_recall_true = best_recall_true[:k]
        idcg = best_recall_true / np.log2(np.arange(1, k + 1) + 1)
        idcg = np.sum(idcg)
        ndcg.append(dcg / idcg)


    precision = np.sum(precision) 
    recall = np.sum(recall) 
    ndcg = np.sum(ndcg)

    return precision, recall, ndcg

def fix_seed(seed=2020):
    """
    固定随机数种子
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dcg(x, name, ascending):
    # 注意y_pred与y_true必须是一一对应的，并且y_pred越大越接近label=1(用相关性的说法就是，与label=1越相关)
    df = pd.DataFrame({"y_pred": x[name], "y_true": x["label"]})
    df = df.sort_values(by='y_pred', ascending=ascending)  # 对y_pred进行降序排列，越排在前面的，越接近label=1
    dcg = df["y_true"] / np.log2(np.arange(1, df["y_true"].count() + 1) + 1)  # 位置从1开始计数
    dcg = np.sum(dcg)
    return dcg


def get_ndcg(x, name, ascending):
    dcg = get_dcg(x, name, ascending)
    if dcg == 0:
        return 0
    idcg = get_dcg(x, "label", False)
    ndcg = dcg / idcg
    return ndcg

def collate_fn(batch_data):
    """This function will be used to pad the graph to max length in the batch
       It will be used in the Dataloader
    """
    uids, iids, labels = [], [], []
    u_items, u_users, u_users_items, i_users = [], [], [], []
    u_items_len, u_users_len, i_users_len = [], [], []

    for uid, iid, label, u_items_u, u_users_u, u_users_items_u, i_users_i in batch_data:

        uids.append(uid)
        iids.append(iid)
        labels.append(label)

        # user-items
        if len(u_items_u) <= truncate_len:
            u_items.append(u_items_u)
        else:
            u_items.append(random.sample(u_items_u, truncate_len))
        u_items_len.append(min(len(u_items_u), truncate_len))

        # user-users and user-users-items
        if len(u_users_u) <= truncate_len:
            u_users.append(u_users_u)
            u_u_items = []
            for uui in u_users_items_u:
                if len(uui) < truncate_len:
                    u_u_items.append(uui)
                else:
                    u_u_items.append(random.sample(uui, truncate_len))
            u_users_items.append(u_u_items)
        else:
            sample_index = random.sample(list(range(len(u_users_u))), truncate_len)
            u_users.append([u_users_u[si] for si in sample_index])

            u_users_items_u_tr = [u_users_items_u[si] for si in sample_index]
            u_u_items = []
            for uui in u_users_items_u_tr:
                if len(uui) < truncate_len:
                    u_u_items.append(uui)
                else:
                    u_u_items.append(random.sample(uui, truncate_len))
            u_users_items.append(u_u_items)

        u_users_len.append(min(len(u_users_u), truncate_len))

        # item-users
        if len(i_users_i) <= truncate_len:
            i_users.append(i_users_i)
        else:
            i_users.append(random.sample(i_users_i, truncate_len))
        i_users_len.append(min(len(i_users_i), truncate_len))

    batch_size = len(batch_data)

    # padding
    # u_items_maxlen = max(u_items_len)
    # u_users_maxlen = max(u_users_len)
    # i_users_maxlen = max(i_users_len)
    u_items_maxlen = truncate_len
    u_users_maxlen = truncate_len
    i_users_maxlen = truncate_len
    u_item_pad = torch.zeros([batch_size, u_items_maxlen, 2], dtype=torch.long)
    for i, ui in enumerate(u_items):
        u_item_pad[i, :len(ui), :] = torch.LongTensor(ui)

    u_user_pad = torch.zeros([batch_size, u_users_maxlen], dtype=torch.long)
    for i, uu in enumerate(u_users):
        u_user_pad[i, :len(uu)] = torch.LongTensor(uu)

    u_user_item_pad = torch.zeros([batch_size, u_users_maxlen, u_items_maxlen, 2], dtype=torch.long)
    for i, uu_items in enumerate(u_users_items):
        for j, ui in enumerate(uu_items):
            u_user_item_pad[i, j, :len(ui), :] = torch.LongTensor(ui)

    i_user_pad = torch.zeros([batch_size, i_users_maxlen, 2], dtype=torch.long)
    for i, iu in enumerate(i_users):
        i_user_pad[i, :len(iu), :] = torch.LongTensor(iu)

    return torch.LongTensor(uids), torch.LongTensor(iids), torch.FloatTensor(labels), \
           u_item_pad, u_user_pad, u_user_item_pad, i_user_pad

