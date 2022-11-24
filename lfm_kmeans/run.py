import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lfm import LFM_algorithm_base_cluster, LFM_Recommand
from metric import Recall_Precison, Coverage, Popularity
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from kmeans import k_means
np.random.seed(1234)

def SampleSelect(data, Popularity, ratio):
    '''
    :param data:用户课程评分数据集
    :param Popularity:字典形式，物品流行度
    :param ratio:负样本比例（负样本：正样本）
    :return:字典形式，用户采样物品。用户id:{物品id:1,..,物品id:0},1代表正样本，0代表负样本
    '''
    Sample = dict()
    popular_items = [i[0] for i in sorted(Popularity.items(), key=lambda x: x[1], reverse=True)]  # 热门商品降序排列
    user_list = set(data['id'])

    for user in user_list:
        PositiveSample = list(data[data['id'] == user]['course_id'])
        N = len(PositiveSample) * ratio
        SelectSample = popular_items[:(N + len(PositiveSample))]
        NegativeSamlpe = [i for i in SelectSample if i not in PositiveSample][:N]  # 得到采样负样本
        positive_di = dict.fromkeys(PositiveSample, 1)  # 正样本标记为1
        negative_di = dict.fromkeys(NegativeSamlpe, 0)  # 负样本标记为0
        positive_di.update(negative_di)
        Sample.setdefault(user, positive_di)  # 把正负样本写进字典
    return Sample

def map_cluster_into_array(user_cluster_ls, cluster_size):
    res = np.zeros(cluster_size, )
    for idx in user_cluster_ls:
        res[idx] = 1
    return res.reshape(1, -1)

def one_hot_encode(feat, n_class):
    res = np.zeros(n_class, )
    for idx in feat:
        res[idx] = 1
    return res.reshape(1, -1)

if __name__ == "__main__":
    MIN_ITEM_COUNT = 100 #过滤掉item出现次数少于MIN_ITEM_COUNT的课程, 设置必须大于1
    MIN_USER_COUNT = 10  #过滤掉user出现次数少于MIN_USER_COUNT的课程, 设置必须大于1
    #lfm的训练参数
    n_clusters = 128 #聚类的个数
    alpha = 0.02 #lfm的学习率,一般不做修改
    lamda = 0.01 #lfm的学习率,一般不做修改
    max_iter=50 #lfm的迭代次数
    need_cluster = True #是否需要聚类

    #差分隐私参数dp
    need_private = True #是否需要在聚类里面引入差分隐私  差分隐私算法见论文https://arxiv.org/abs/2010.01234.
    privacy_budget = 320 #这个数要参考user聚类的维度, 如果该数越小, 在差分隐私里加入的噪声越大

    user_feat_name = ["id", "teacher", "school"]
    user_item_feat_name = ["id", "course_id"]

    data = pd.read_csv("./data/data_new_1028.csv", encoding = "ISO-8859-1", index_col=0)
    need_del_item_data = data.groupby("course_id").filter(lambda x: len(x) < MIN_ITEM_COUNT)
    need_del_user_data = data.groupby("id").filter(lambda x: len(x) < MIN_USER_COUNT)
    need_del_data = need_del_item_data.append(need_del_user_data)
    data = data.drop(need_del_data.index).reset_index()
    data["rate"] = 1   #所有的user-course的标签都设为1

    #对样本的id类特征进行encode
    for feat_name in user_feat_name + user_item_feat_name[1: ]:
        lbe = LabelEncoder()
        data[feat_name] = lbe.fit_transform(data[feat_name])

    # 物品流行度：有多少用户为某物品评分--主要用于负样本生成
    item_count = data['course_id'].value_counts()
    item_popularity = dict(zip(item_count.index, item_count.values))  # 物品流行度字典,物品id:流行度

    # 训练测试集划分
    #先把userid count=1的划分给训练集
    train = data.groupby("id").filter(lambda x: len(x) < 2)
    # next_data = data.drop(train.index)
    # train = train.append(next_data.groupby("course_id").filter(lambda x: len(x) < 2))
    train_others, test = train_test_split(data.drop(train.index), random_state=1234, shuffle=True, test_size=0.2, stratify=data.drop(train.index)["id"])
    train = train.append(train_others)
    user_train = {}
    user_test = {}
    for user in set(test['id']):
        user_test[user] = list(test[test['id'] == user]['course_id'])

    M, N = len(set(train["id"].values)), len(set(train["course_id"].values))
    print("the user number is: {}, the item number is: {}".format(M, N))

    user_cluster = None
    if need_cluster:

        print ("**********start cluster for user by kmeans*******")
        #构建聚类的特征,每一列特征都用one-hot编码
        train_user_feat = train[user_feat_name]
        train_user_feat_group = train_user_feat.groupby("id")
        user_feat_input = []
        for idx, feat_name in enumerate(user_feat_name[1:]):
            feat_df = train_user_feat_group[feat_name].apply(lambda x: one_hot_encode(set(x), train_user_feat[feat_name].max()+1)).reset_index()
            user_feat_input.append(np.concatenate(feat_df[feat_name].values, axis=0))
        user_feat_input_array = np.concatenate(user_feat_input, axis=1)

        if need_private:
            user_cluster, _ = k_means(user_feat_input_array, n_clusters, privacybudget=privacy_budget)
        else:
            user_cluster, _ = k_means(user_feat_input_array, n_clusters)
        # user_cluster = KMeans(n_clusters=n_clusters, init='k-means++', random_state=1234).fit_transform(user_feat_input_array)
        print("**********done cluster for user by kmeans, shape: ({}, {})*******".format(user_cluster.shape[0], user_cluster.shape[1]))
    else:
        user_cluster = np.random.rand(M, n_clusters)
    LFM_li = []
    # for ratio in [1,2,3,5,10,20]:
    for ratio in [5]:
        user_recommond = dict()
        Sample = SampleSelect(train, item_popularity, ratio)
        t = time.time()

        #将聚类结果替换lfm算法里面的p
        R_train = LFM_algorithm_base_cluster(Sample, M, N, user_cluster, alpha, lamda, max_iter=max_iter)
        print('ratio {} cost time:{:.4}s'.format(ratio, (time.time() - t)))
        test["pred_score"] = test.apply(lambda x: R_train[0].loc[x["id"], x["course_id"]], axis=1)
        mae = mean_absolute_error(test["rate"].values, test["pred_score"].values)
        rmse = mean_squared_error(test["rate"].values, test["pred_score"].values)
        for user in list(set(test['id'])):
            user_recommond[user] = LFM_Recommand(user, R_train[0], 20)

        r =  Recall_Precison(user_recommond, user_test)   #召回率和精确率
        cov = Coverage(user_recommond, len(set(test['course_id']))) #覆盖率
        p = Popularity(user_recommond, item_popularity)  #推荐流行度计算
        print ("mae: {:.4f}, rmse: {:.4f},  precision: {:.4f},  recall: {:.4f}, coverage: {:.4f}, popularity: {:.4f}".format(mae, rmse, r[0], r[1], cov, p))
        LFM_li.append([ratio, mae, rmse, r[0], r[1], cov, p])

    LFM = pd.DataFrame(LFM_li, columns=['ratio', 'mae', 'rmse', 'precision', 'recall', 'coverage', 'popularity'])
    # print (LFM)