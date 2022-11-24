from time import time
from typing import ItemsView
import torch
from torch import nn
from torch_geometric.nn import GatedGraphConv
from torch_geometric.utils import softmax
from torch_scatter import scatter_sum


class Model(nn.Module):
    def __init__(self,args, item_num, user_num) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.n_item = item_num
        self.n_entity = user_num

        self.user_embedding = nn.Embedding(user_num+1, self.hidden_size, padding_idx=-1)
        self.item_embedding = nn.Embedding(item_num, self.hidden_size)
        self.rating_embedding = nn.Embedding(10,self.hidden_size)
        
        self.gv = nn.Linear(self.hidden_size*2, self.hidden_size)

        self.gu=nn.Linear(self.hidden_size*2, self.hidden_size)

        self.eq5 = nn.Sequential(
                    nn.Linear(self.hidden_size*2,self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size,1)
        )
        
        self.eq5i = nn.Sequential(
                    nn.Linear(self.hidden_size*2,self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size,1)
        )
        
        self.eq4 = nn.Sequential(
                    nn.Linear(self.hidden_size,self.hidden_size),
                    nn.ReLU()
        )
        
        self.eq4i = nn.Sequential(
                    nn.Linear(self.hidden_size,self.hidden_size),
                    nn.ReLU()
        )

        self.eq10 = nn.Sequential(
                    nn.Linear(self.hidden_size*2,self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size,1)
        )
        self.eq9 = nn.Sequential(
                    nn.Linear(self.hidden_size,self.hidden_size),
                    nn.ReLU()
        )

        self.eq13 = nn.Sequential(
                    nn.Linear(self.hidden_size*2, self.hidden_size),
                    nn.ReLU()
        )
        
        
        self.mlp=nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size, bias = True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias = True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1, bias = True)
        )

        self.loss_function = nn.MSELoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.1
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, user,item,rating,item4user,social, social4user, batch_i_users,batch_i_ratings,batch_u_item,batch_target):
    
        user_emb = self.user_embedding(user) # pi
        item_emb = self.item_embedding(item) # qj #用户交替item
        rating_emb = self.rating_embedding(rating) # er
        
        

        social_emb = self.user_embedding(social)
        
        

        # item aggregation
        xia = self.gv(torch.concat([item_emb,rating_emb],-1)) # eq 2
        
       

        alpha_ia = self.eq5(torch.concat([xia, user_emb[item4user]], -1))  # eq 5 两个大小必须一样才能横着拼接

        alpha_ia = softmax(alpha_ia, item4user)  # eq 6 相同index元素softmax

        hi = self.eq4(scatter_sum((xia * alpha_ia), item4user, dim=0))  # eq4
        # 将index相同值对应的src元素进行对应定义的计算
        
        
        item_u_emb=self.user_embedding(batch_i_users)
        item_i_rating=self.rating_embedding(batch_i_ratings)
        
        

        fjt = self.gu(torch.concat([item_u_emb,item_i_rating],-1)) # eq 2
        ujt = self.eq5i(torch.concat([fjt,self.item_embedding(batch_target)[batch_u_item]],-1)) # eq 5
        
        ujt = softmax(ujt, batch_u_item) # eq 6
        
        # print((fjt*ujt).shape)
        # print(batch_u_item.shape)
        zj = self.eq4i(scatter_sum((fjt*ujt),batch_u_item,dim=0)) # eq4
        
        
        
        
        
        

        

       
        
        
        
        
        
        


         
        
        
        #alpha_ia = self.eq5(torch.concat([xia,user_emb],0)) # eq 5

        #alpha_ia = torch.nn.functional.softmax(alpha_ia,dim=0) # eq 6
        
       
        #hi = self.eq4(torch.sum(alpha_ia*xia,0)) # eq4
        
        
        
       

        # social aggregation

        beta_io = self.eq10(torch.concat([social_emb,user_emb[social4user]],-1)) # eq 5
        beta_io = softmax(beta_io, social4user) # eq 11
        hs =  self.eq9(scatter_sum((user_emb[social4user]*beta_io),social4user,dim=0)) # eq9
        
        #beta_io = self.eq10(torch.concat([social_emb,user_emb[social4user]],-1)) # eq 5
        #beta_io = torch.nn.functional.softmax(beta_io, dim=0) # eq 11
        #hs =  self.eq9(torch.sum(beta_io*user_emb[social4user],0)) # eq9
        
        

        # learning user latent factor

        h = self.eq13(torch.concat([hi,hs],-1))
        
       
        
        
        
        return self.mlp(torch.concat([h,zj],dim=-1)), torch.matmul(h,self.item_embedding.weight.T)
        
        
        
        

        #recommendation
        
        # return torch.matmul(h,self.item_embedding.weight.T)
