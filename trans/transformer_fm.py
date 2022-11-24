from sklearn.preprocessing import LabelEncoder
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import math
# import torchme
from urllib.request import urlretrieve
from zipfile import ZipFile
import os
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import functools
from sklearn.metrics import roc_auc_score

import pandas as pd
import torch
import torch.utils.data as data
from activation import activation_layer
from sklearn import metrics

users = pd.read_csv(
    "data/users.csv",
    sep=",",
)
ratings = pd.read_csv(
    "data/ratings.csv",
    sep=",",
)

# ratings_X = ratings.iloc[:, :]
# ratings_y = ratings.rating.values
# ratings_X = ratings_X.apply(LabelEncoder().fit_transform)
users_X = users.iloc[:, 0:4]
# ratings_y = ratings.rating.values
users_X = users_X.apply(LabelEncoder().fit_transform)
field_dim = 4 # 模型输入的feature_fields
# print("field_dim::",field_dim)
# print("user:",users_X.shape)
EMBEDDING_DIM = 8

movies = pd.read_csv(
    "data/movies.csv", sep=","
)



class MovieDataset(data.Dataset):
    """Movie dataset."""

    def __init__(
            self, ratings_file, test=False
    ):
        """
        Args:
            csv_file (string): Path to the csv file with user,past,future.
        """
        self.ratings_frame = pd.read_csv(
            ratings_file,
            delimiter=",",
            # iterator=True,
        )
        self.test = test

        # self.items = data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        # self.field_dims = np.max(self.items, axis=0) + 1
        """
            a=np.array([[378., 533.],
               [456., 420.],
               [593., 461.],
               [529., 584.]])
            np.min(a)
            [output]:378.
            
            np.min(a,axis=1)
            [output]:array([378., 420., 461., 529.])
            
            np.max(),np.min() 返回数组中所有数据中的最大值或最小值
            加入axis参数，当axis=0时会分别取每一列的最大值或最小值，axis=1时，会分别取每一行的最大值或最小值，且将所有取到的数据放在一个一维数组中。
        """
        # 为什么要+1?
        # self.field_dims = np.max(self.items, axis=0) + 1

    def __len__(self):
        return len(self.ratings_frame)

    def __getitem__(self, idx):
        data = self.ratings_frame.iloc[idx]

        user_id = data.user_id

        movie_history = eval(data.sequence_movie_ids)
        movie_history_ratings = eval(data.sequence_ratings)
        target_movie_id = movie_history[-1:][0]
        target_movie_rating = movie_history_ratings[-1:][0]

        movie_genres_history = torch.LongTensor(movies[movies.movie_id.isin(movie_history[:-1])][genres].to_numpy())
        movie_year_history = torch.LongTensor(movies[movies.movie_id.isin(movie_history[:-1])]["year"].to_numpy())
        movie_history = torch.LongTensor(movie_history[:-1])
        movie_history_ratings = torch.LongTensor(movie_history_ratings[:-1])

        sex = data.sex
        age_group = data.age_group
        occupation = data.occupation

        return user_id, movie_history, movie_genres_history, movie_year_history, target_movie_id, movie_history_ratings, target_movie_rating, sex, age_group, occupation


genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

for genre in genres:
    movies[genre] = movies["genres"].apply(
        lambda values: int(genre in values.split("|"))
    )

sequence_length = 8

train_dataset = MovieDataset("data/train_data.csv")
val_dataset = MovieDataset("data/test_data.csv")
test_dataset = MovieDataset("data/test_data.csv")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=16)
valid_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=8)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=8)


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


# class FactorizationMachine(nn.Module):
#     """
#         Factorization Machine
#     """
#
#     def __init__(self, feature_fields, embed_dim):
#         """
#             feature_fileds : array_like
#                              类别特征的field的数目
#         """
#         super(FactorizationMachine, self).__init__()
#
#         # 输入的是label coder 用输出为1的embedding来形成linear part
#         self.linear = torch.nn.Embedding(sum(feature_fields) + 1, 1)
#         self.bias = torch.nn.Parameter(torch.zeros((1,)))
#
#         self.embedding = torch.nn.Embedding(sum(feature_fields) + 1, embed_dim)
#         self.offset = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype=np.long)
#         nn.init.xavier_uniform_(self.embedding.weight.data)
#
#     def forward(self, x):
#         # print("x",x)
#         print("x.shape",x.shape)
#         y = x.new_tensor(self.offset).unsqueeze(0)
#         # print("y",y)
#         print("y.shape", y.shape)
#         tmp = x + y
#         print("tmp.shape", tmp.shape)
#         # 线性层
#         linear_part = torch.sum(self.linear(tmp), dim=1) + self.bias
#         # 内积项
#         ## embedding
#         tmp = self.embedding(tmp)
#         ##  XY
#         square_of_sum = torch.sum(tmp, dim=1) ** 2
#         sum_of_square = torch.sum(tmp ** 2, dim=1)
#
#         x = linear_part + 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
#         # sigmoid
#         x = torch.sigmoid(x.squeeze(1))
#         return x

class DNN(nn.Module):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **inputs_dim**: input feature dimension.
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class FM(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term

class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.
    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class BST(nn.Module):
    def __init__(self, args=None):
        super(BST, self).__init__()

        # self.save_hyperparameters()
        self.args = args
        # -------------------
        # Embedding layers
        ##Users
        self.embeddings_user_id = nn.Embedding(
            int(users.user_id.max()) + 1, field_dim
        )
        ###Users features embeddings
        self.embeddings_user_sex = nn.Embedding(
            len(users.sex.unique()), field_dim
        )
        self.embeddings_age_group = nn.Embedding(
            len(users.age_group.unique()), field_dim
        )
        self.embeddings_user_occupation = nn.Embedding(
            len(users.occupation.unique()), field_dim
        )
        self.embeddings_user_zip_code = nn.Embedding(
            len(users.zip_code.unique()), field_dim
        )

        ##Movies
        self.embeddings_movie_id = nn.Embedding(
            int(movies.movie_id.max()) + 1, int(math.sqrt(movies.movie_id.max())) + 1
        )
        self.embeddings_position = nn.Embedding(
            sequence_length, int(math.sqrt(len(movies.movie_id.unique()))) + 1
        )
        ###Movies features embeddings
        genre_vectors = movies[genres].to_numpy()
        self.embeddings_movie_genre = nn.Embedding(
            genre_vectors.shape[0], genre_vectors.shape[1]
        )
        self.embeddings_movie_genre = nn.Embedding(genre_vectors.shape[1], 4)
        # self.embeddings_movie_genre.weight.requires_grad = False  # Not training genres

        self.embeddings_movie_year = nn.Embedding(
            len(movies.year.unique()), field_dim
        )

        #item侧特征=(genre + year) * senquence_length * field_dim + user_feature size * field_dim
        self.dnn = DNN((genre_vectors.shape[1] + 1) * 7 * field_dim + 4 * field_dim, (EMBEDDING_DIM * 2, EMBEDDING_DIM))
        self.dnn_linear = nn.Linear(EMBEDDING_DIM, 1, bias=False)
        self.fm = FM()

        # Network
        self.transfomerlayer = nn.TransformerEncoderLayer(63, 3, dropout=0.5)
        self.linear = nn.Sequential(
            nn.Linear(
                505,
                1024,
            ),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # self.criterion = torch.nn.MSELoss()
        # self.mae = torchmetrics.MeanAbsoluteError()
        # self.mse = torchmetrics.MeanSquaredError()

    def encode_input(self, inputs):
        user_id, movie_history, movie_genres_history, movie_year_history, target_movie_id, movie_history_ratings, target_movie_rating, sex, age_group, occupation = inputs

        # MOVIES
        movie_genre_history = self.embeddings_movie_genre(movie_genres_history)
        movie_genre_history = movie_genre_history.reshape(movie_genre_history.shape[0], movie_genre_history.shape[1] * movie_genre_history.shape[2], -1)

        movie_year_history = self.embeddings_movie_year(movie_year_history)
        movie_feat = torch.cat([movie_genre_history, movie_year_history], axis=1)

        movie_history = self.embeddings_movie_id(movie_history)
        target_movie = self.embeddings_movie_id(target_movie_id)

        positions = torch.arange(0, sequence_length - 1, 1, dtype=int).cuda()
        # positions = torch.arange(0, sequence_length - 1, 1, dtype=int)
        positions = self.embeddings_position(positions)

        encoded_sequence_movies_with_poistion_and_rating = (movie_history + positions)  # Yet to multiply by rating

        target_movie = torch.unsqueeze(target_movie, 1)
        transfomer_features = torch.cat((encoded_sequence_movies_with_poistion_and_rating, target_movie), dim=1)

        # USERS
        user_id = self.embeddings_user_id(user_id).unsqueeze(1)
        sex = self.embeddings_user_sex(sex).unsqueeze(1)
        age_group = self.embeddings_age_group(age_group).unsqueeze(1)
        occupation = self.embeddings_user_occupation(occupation).unsqueeze(1)
        user_features = torch.cat([user_id, sex, age_group, occupation], 1)
        fm_features = torch.cat([user_features, movie_feat], axis=1)
        return transfomer_features, fm_features, target_movie_rating

    def forward(self, batch):
        transfomer_features, fm_features, target_movie_rating = self.encode_input(batch)

        target_movie_rating[target_movie_rating <= 3] = 0
        target_movie_rating[target_movie_rating > 3] = 1
        # target_movie_rating = nn.functional.one_hot(target_movie_rating.long())

        transformer_output = self.transfomerlayer(transfomer_features)
        transformer_output = torch.flatten(transformer_output, start_dim=1)

        #fm包括linear侧dnn跟特征交叉侧fm
        dnn_input = fm_features.reshape(fm_features.shape[0], -1)
        dnn_output = self.dnn(dnn_input) #(bs, EMBEDDING_DIM)

        dnn_logit = self.dnn_linear(dnn_output)  #(bs, 1)
        fm_logit = self.fm(fm_features)  #(bs, 1)
        dnn_logit += fm_logit
        # Concat with other features
        features = torch.cat((transformer_output, dnn_logit), dim=1)

        output = self.linear(features)
        # output = self.softmax(output)
        return output, target_movie_rating


# model = 'state_dict_epoch:1.pth'
BST = BST()
# BST.load_state_dict(torch.load(model))
BST = BST.cuda()


optimizer = torch.optim.Adam(BST.parameters(), lr=0.0001, eps=1e-3)
# criterion = torch.nn.BCEWithLogitsLoss().cuda()
criterion = FocalLoss()
epochs = 50

# training
for epoch in range(epochs):
    print("----------------------epoch:{:d}----------------------\n".format(epoch))
    torch.cuda.synchronize()
    BST.train()
    batch_loss = 0.0
    batch_accuracy = 0.0
    batch_auc = 0.0

    pos_num = 0
    neg_num = 0

    for i, input_dict in enumerate(tqdm(train_loader, 0)):
        for k, v in enumerate(input_dict):
            input_dict[k] = Variable(v, requires_grad=False).cuda()
            # input_dict[k] = Variable(v, requires_grad=False)
        optimizer.zero_grad()
        output, target_movie_rating = BST.forward(input_dict)
        fpr, tpr, thresholds = metrics.roc_curve(target_movie_rating.cpu().detach().numpy(), output.squeeze(-1).cpu().detach().numpy())
        auc = metrics.auc(fpr, tpr)
        output = torch.clamp_min(output, 1e-6)
        pos_num += torch.sum(target_movie_rating)
        neg_num += target_movie_rating.shape[0] - torch.sum(target_movie_rating)

        # output = output.flatten()
        #         # print(output, target_movie_rating.long())
        #         # print("target_movie_rating",target_movie_rating)
        #         # print(type(out))
        #         # print(type(target_movie_rating))
        #         # print(output)
        loss = criterion(output.squeeze(-1), target_movie_rating.float())
        # print(loss.data)
        # output = torch.argmax(output, dim=1)

        loss.backward()
        clip_grad_norm_(BST.parameters(), max_norm=3, norm_type=2)
        optimizer.step()
        torch.cuda.synchronize()
        output.detach()
        output[output > 0.5] = 1
        output[output <= 0.5] = 0

        true = torch.sum(output.squeeze(-1) == target_movie_rating).float()
        # print(true)
        accuracy = true / target_movie_rating.shape[0]

        batch_loss += loss.data
        batch_accuracy += accuracy
        batch_auc += auc

        if (i + 1) % 100 == 0:
            print('\n ---- batch: %03d ----' % (i + 1))
            print('loss:{:f}, accuracy:{:f}'.format(batch_loss / 100, batch_accuracy / 100))
            batch_loss = 0.0
            batch_accuracy = 0.0
            batch_auc = 0.0
    print(pos_num, neg_num)

    if (epoch + 1) % 10 == 0:
        torch.save(BST.state_dict(), 'state_dict_epoch:%d.pth' % (epoch))


# validation


# torch.cuda.synchronize()
# BST.eval()
# batch_loss = 0.0
# for i, input_dict in enumerate(tqdm(valid_loader, 0)):
#     for k, v in enumerate(input_dict):
#         input_dict[k] = Variable(v, requires_grad=False).cuda()
#
#     output, target_movie_rating = BST.forward(input_dict)
#     output = output.flatten()
#     # print("out:",out)
#     # print("target_movie_rating",target_movie_rating)
#     # print(type(out))
#     # print(type(target_movie_rating))
#     loss = criterion(output, target_movie_rating)
#     torch.cuda.synchronize()
#
#     batch_loss += loss.data
#     if (i + 1) % 100 == 0:
#         print('\n ---- batch: %03d ----' % (i + 1))
#         print('loss:{:f}'.format(batch_loss / 100))
#         batch_loss=0.0
#     # print(output, target_movie_rating)
#     # auc = caculateAUC(output, target_movie_rating)
