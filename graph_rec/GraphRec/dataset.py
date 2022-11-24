import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import collate_fn
import os
import pandas as pd 
from collections import defaultdict
import pickle

def load_data(args):
    dataset_folder = os.path.abspath(os.path.join('dataset'))

    with open(os.path.join(dataset_folder,'data.txt'),'r') as reader :
        line = reader.readline().strip().split(' ')
        user_num,item_num = eval(line[0]) , eval(line[1])

    train_set = MyDataset(dataset_folder,'train2')

    test_l_set = MyDataset(dataset_folder,'test2',l=True) # 冷启动数据

    test_set = MyDataset(dataset_folder,'test2')

    train_loader = DataLoader(train_set,args.batch_size,  num_workers=0,
                              shuffle=True,collate_fn=collate_fn,drop_last=True)
    test_l_loader = DataLoader(test_l_set,args.test_size, num_workers=0,
                              shuffle=False,collate_fn=collate_fn)
    test_loader = DataLoader(test_set,args.test_size, num_workers=0,
                              shuffle=False,collate_fn=collate_fn)

    
    all_data = train_set.data + test_l_set.data + test_set.data
    all_data = pd.DataFrame(all_data, columns=["userid", "itemid", "rate"])
    return train_loader, test_loader,test_l_loader, item_num, user_num, all_data.sort_values(axis = 0, ascending = True, by = 'userid')
    


class MyDataset(Dataset):
    def __init__(self, datafolder, file,l=False) -> None:
        super().__init__()
        
        if file=='test2':
        
            data_df = pd.read_csv(os.path.join(datafolder,'test2'+'.txt'),names=['user','item','score','time'],sep=' ',dtype='int')
        else:
            data_df = pd.read_csv(os.path.join(datafolder,'train2'+'.txt'),names=['user','item','score','time'],sep=' ',dtype='int')
       
        train_df = pd.read_csv(os.path.join(datafolder,'train2'+'.txt'),names=['user','item','score','time'],sep=' ',dtype='int')

        self.user_item = defaultdict(list) # user交互的列表

        self.trust = defaultdict(list) # user社交网络列表
        
        
        self.item_user=defaultdict(list) # item交互的列表

        with open(os.path.join(datafolder,'trust.txt'),'r') as reader:
            lines = reader.readlines()
            for line in lines:
                users = line.strip().split(' ')
                self.trust[int(users[0])].append(int(users[1]))

        self.data = []
        for line in train_df.to_numpy():
            self.user_item[int(line[0])].append((int(line[1]),int(line[2]))) # item score
            self.item_user[int(line[1])].append((int(line[0]),int(line[2])))

        for line in data_df.to_numpy():
            if l: # 如果是冷启动数据集 
                if len(self.user_item[line[0]]) > 5:
                    continue
            self.data.append([int(line[0]),int(line[1]),int(line[2])]) # 训练数据 user item




    def __len__(self):
        return len(self.data)

    def __getitem__(self, index): 
        userid = self.data[index][0]
        target = self.data[index][1]
        labels = self.data[index][2]
        
        
        

        items_ratings = []

        users_ratings=[]#商品的用户
        #若商品不存在user,则设置默认值，这里给user size，注意要和item embedding里设置padding_idx对应
        if len(self.user_item[userid]) == 0:
            items_ratings.append((16861, 0))
        else:
            for i,r in self.user_item[userid]:
                items_ratings.append((i, r))

        #若user不存在user,则设置默认值，这里给user size，注意要和user embedding里设置padding_idx对应
        if len(self.item_user[target]) == 0:
            users_ratings.append((2378, 0))
        else:
            for u,r in self.item_user[target]:
                    users_ratings.append((u, r))
        
        social = self.trust[userid]
        social_items_ratings = []
        if len(social) == 0:
            social = [2378]
            social_items_ratings = [[(16861, 0)]]
        else:
            for u_user in social:
                tmp_items_ratings = []
                # 若商品不存在user,则设置默认值，这里给user size，注意要和item embedding里设置padding_idx对应
                if len(self.user_item[u_user]) == 0:
                    tmp_items_ratings.append((16861, 0))
                else:
                    for i, r in self.user_item[u_user]:
                        tmp_items_ratings.append((i, r))
                social_items_ratings.append(tmp_items_ratings)


        return userid, target, labels, items_ratings, social, social_items_ratings, users_ratings


# def collate_fn(batch_data):
#
#     batch_user = [] # 用户id
#     batch_y = [] # 标签
#
#     batch_items = [] # 所有用户的交互item
#     batch_ratings = [] # rating
#     batch_item4user = []  # 每个item对应的用户
#
#     batch_socials = [] # 所有用户的trust
#     batch_social4user = []  # 每个trust对应的user
#
#     batch_i_users,batch_i_ratings=[],[]
#     batch_u_item=[]
#
#
#     batch_target=[]
#
#
#
#
#
#     for idx,data in enumerate(batch_data):
#
#         userid = data[0]
#         target = data[1]
#         labels=data[2]
#         items = data[3]
#         ratings = data[4]
#         socials = data[5]
#
#         i_users=data[6]
#         i_ratings=data[7]
#
#         batch_i_users.extend(i_users)
#         batch_i_ratings.extend(i_ratings)
#
#         batch_target.append(target)
#
#         batch_u_item.extend([idx]*len(i_users))
#
#         #测试时460 8161 1023 0 2466处报错
#
#
#
#
#         batch_user.append(userid)
#         batch_y.append(labels)
#
#         batch_items.extend(items)
#         batch_ratings.extend(ratings)
#         batch_item4user.extend([idx]*len(items)) #这两个就是为了标记batch_items和batch_social中的每个值属于batch中的哪一个用户
#
#         if len(socials) == 0:
#             socials = [userid]
#         batch_socials.extend(socials)
#         batch_social4user.extend([idx]*len(socials))
#
#     batch_user = torch.LongTensor(batch_user)
#     batch_y = torch.LongTensor(batch_y)
#
#     batch_items = torch.LongTensor(batch_items)
#     batch_ratings = torch.LongTensor(batch_ratings)
#     batch_item4user = torch.LongTensor(batch_item4user)
#
#     batch_socials = torch.LongTensor(batch_socials)
#     batch_social4user = torch.LongTensor(batch_social4user)
#
#     batch_i_users = torch.LongTensor(batch_i_users)
#     batch_i_ratings = torch.LongTensor(batch_i_ratings)
#     batch_u_item = torch.LongTensor(batch_u_item)
#
#     batch_target=torch.LongTensor(batch_target)
#
#     return batch_user, batch_y, batch_items, batch_ratings, batch_item4user, batch_socials, batch_social4user, batch_i_users, batch_i_ratings, batch_u_item, batch_target
