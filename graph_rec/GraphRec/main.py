from unittest import TestLoader
from tqdm import tqdm
from dataset import load_data

import torch
import time
from parse import get_parse
from utils import fix_seed, topk_recall_precision_ndgc
from model import Model
from model_bak import GraphRec
from sklearn import metrics


import torch

if __name__ == '__main__':

    args = get_parse()

    fix_seed()

    train_loader, test_loader,test_l_loader, item_num, user_num, user_item_rate_df = load_data(args)

    



    model = GraphRec(user_num+1, item_num+1, 5+1, args.hidden_size)
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   

    e_stop = 0
    precision_max,recall_max,ndgc_max = {k:0 for k in args.topk}, {k:0 for k in args.topk},{k:0 for k in args.topk} # 记录最高的topk分数
    precision_l_max,recall_l_max,ndgc_l_max = {k:0 for k in args.topk}, {k:0 for k in args.topk},{k:0 for k in args.topk} # 记录最高的topk分数

    item_embedding_table = [torch.zeros((1, 30, 2), dtype=(torch.int64)) for i in range(item_num)]
    user_item_rating = []
    for e in range(args.epoch):
        # rec train
        model.train()
        all_loss = 0.0

        #将item embedding需要保存下来, 测试时用于recall
        for user, item, label, items_rating, social, social_items_ratings, i_users in train_loader:
            if e == 0:
                for i, iid in enumerate(item.numpy()):
                    item_embedding_table[iid] = i_users[i].unsqueeze(0)
            output, _ = model(user.to(args.device),
                            item.to(args.device),
                            items_rating.to(args.device),
                            social.to(args.device),
                            social_items_ratings.to(args.device),
                            i_users.to(args.device))
            optimizer.zero_grad() 
            
            loss = torch.nn.MSELoss()(output, label.to(torch.float32).unsqueeze(1)) # 因为没有测试pad
            loss.backward()
            optimizer.step()
            all_loss += loss.item()

        # 预测前需要更新item侧embedding,输出所有item的结果
        item_embedding_tensor = model.get_item_embedding(torch.range(0, item_num-1, dtype=torch.int64).to(args.device), torch.cat(item_embedding_table, axis=0).to(args.device))
        if e%5==1:
            # 测试模型
            model.eval()
            precision, recall,ndgc = dict((k,0.0) for k in args.topk), dict((k,0.0) for k in args.topk),dict((k,0.0) for k in args.topk) # 当前epoch每个指标的分数
            test_num = 0

            for user, item, label, items_rating, social, social_items_ratings, i_users in test_loader:
                output, recall_res = model(user.to(args.device),
                                    item.to(args.device),
                                    items_rating.to(args.device),
                                    social.to(args.device),
                                    social_items_ratings.to(args.device),
                                    i_users.to(args.device))

                #这里需要batch_size里面的user跟所有的item embedding计算rating再继续后面的topk召回结果
                item_recall_score = model.item_recall(recall_res, item_embedding_tensor)
                for k in precision.keys():
                    this_hr, this_mrr,this_ndgc = topk_recall_precision_ndgc(user.numpy(), item.numpy(), label.numpy(), item_recall_score.detach().cpu(), user_item_rate_df, k)
                    precision[k] += this_hr
                    recall[k] += this_mrr
                    ndgc[k] += this_ndgc
                test_num += item.shape[0]


            # 冷启动
            precision_l, recall_l,ndgc_l = dict((k,0.0) for k in args.topk), dict((k,0.0) for k in args.topk),dict((k,0.0) for k in args.topk) # 当前epoch每个指标的分数
            test_num = 0
            for user, item, label, items_rating, social, social_items_ratings, i_users in test_loader:
                output, recall_res = model(user.to(args.device),
                                    item.to(args.device),
                                    items_rating.to(args.device),
                                    social.to(args.device),
                                    social_items_ratings.to(args.device),
                                    i_users.to(args.device))
                item_recall_score = model.item_recall(recall_res, item_embedding_tensor)
                for k in precision.keys():
                    this_hr, this_mrr,this_ndgc = topk_recall_precision_ndgc(user.numpy(), item.numpy(), label.numpy(), item_recall_score.detach().cpu(), user_item_rate_df, k)
                    precision_l[k] += this_hr
                    recall_l[k] += this_mrr
                    ndgc_l[k] += this_ndgc
                test_num += item.shape[0]
            print('---result------')
            for k in precision.keys():
                precision[k] /= test_num
                recall[k] /= test_num
                ndgc[k] /= test_num
                print('Precision@%d %.6f'%(k, precision[k]))
                print('Recall@%d %.6f'%(k, recall[k]))
                print('NDcg@%d %.6f'%(k, ndgc[k]))
                #if precision_max[k] < precision[k]: precision_max[k] = precision[k]
                #if recall_max[k] < recall[k]:  recall_max[k] = recall[k]
                if k==10 and ndgc_max[k] < ndgc[k]:
                    e_stop += 1
                else:
                    e_stop = 0
                if ndgc_max[k] < ndgc[k]: 
                   ndgc_max[k] = ndgc[k]
                   

            
            print("----冷启动----")
            for k in precision_l.keys():
                precision_l[k] /= test_num
                recall_l[k] /= test_num
                ndgc_l[k] /= test_num
                print('Precision@%d %.6f'%(k, precision_l[k]))
                print('recall@%d %.6f'%(k, recall_l[k]))
                print('NDcg@%d %.6f'%(k, ndgc_l[k]))
                #if precision_l_max[k] < precision_l[k]: precision_l_max[k] = precision_l[k]
                #if recall_l_max[k] < recall_l[k]:   recall_l_max[k] = recall_l[k]
                #if ndgc_l_max[k] < ndgc_l[k]:   ndgc_l_max[k] = ndgc_l[k]
            print("\n")


            if e_stop >= args.patience:
                print("========Best Score===========")
                for k in recall.keys(): 
                    print('Precision@%d %.6f'%(k, precision_max[k]))
                    print('recall@%d %.6f'%(k, recall_max[k])) 
                    print('NDGC@%d %.6f'%(k, ndgc_max[k])) 

                print("========冷启动 Best Score===========")
                for k in recall.keys(): 
                    print('Precision@%d %.6f'%(k, precision_l_max[k]))
                    print('recall@%d %.6f'%(k, recall_l_max[k])) 
                    print('NDGC@%d %.6f'%(k, ndgc_l_max[k])) 
                break


