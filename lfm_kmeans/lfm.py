import math

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

def LFM_algorithm(Sample, M, N, K, alpha, lamda, max_iter=1000):
    '''
    :param Sample: 采样集合，字典形式
    :param M: 用户size
    :param N: 物品size
    :param K: 隐类别参数
    :param alpha: 学习速率
    :param lamda: 正则化参数
    :param max_iter: 最大迭代次数
    :return:
    用户特征矩阵P(用户与k隐类的关系),规模M*K;
    物品特征矩阵Q(k隐类与物品的关系),规模K*N
    '''
    # 随机初始化特征矩阵P、Q
    P = np.random.rand(M, K)
    Q = np.random.rand(K, N)

    for n in range(max_iter):
        print ("********* current iter {} ***************".format(n))

        for u in range(M):
            for i, label in sorted(Sample[u].items(), key=lambda x: x[0]):
                eui = label - 1 / (1 + np.exp(-np.dot(P[u, :], Q[:, i])))
                # 梯度下降更新Pu,Qi
                P[u, :] = P[u, :] + alpha * (Q[:, i] * eui - lamda * P[u, :])
                Q[:, i] = Q[:, i] + alpha * (P[u, :] * eui - lamda * Q[:, i])



        alpha *= 0.9  # 每次迭代降低学习率
        # # 计算损失函数值
        cost = 0
        # for u in range(M):
        #     for i in range(N):
        #         if Sample[user_list[u]].get(item_list[i], -1) != -1:
        #             cost += (R[u, i] - 1 / (1 + np.exp(-np.dot(P[u, :], Q[:, i])))) ** 2
    # for k in range(K): #计算正则项
    # cost += lamda*(P[u,k]**2 + Q[k,i]**2)

    # if cost < 0.1*:
    # break

    R_new = pd.DataFrame(1 / (1 + np.exp(-np.dot(P, Q))), index=np.arange(M), columns=np.arange(N))

    return R_new, cost

def LFM_algorithm_base_cluster(Sample, M, N, user_cluster, alpha, lamda, max_iter=1000):
    '''
    :param Sample: 采样集合，字典形式
    :param M: 用户size
    :param N: 物品size
    :param user_cluster: 用户的聚类结果
    :param alpha: 学习速率
    :param lamda: 正则化参数
    :param max_iter: 最大迭代次数
    :return:
    用户特征矩阵P(用户与k隐类的关系),规模M*K;
    物品特征矩阵Q(k隐类与物品的关系),规模K*N
    '''
    K = user_cluster.shape[-1]
    # 随机初始化特征矩阵P、Q
    P = user_cluster
    Q = np.random.rand(K, N)

    for n in range(max_iter):
        print ("********* current iter {} ***************".format(n))

        for u in range(M):
            for i, label in sorted(Sample[u].items(), key=lambda x: x[0]):
                eui = label - 1 / (1 + np.exp(-np.dot(P[u, :], Q[:, i])))
                # 梯度下降更新Pu,Qi
                P[u, :] = P[u, :] + alpha * (Q[:, i] * eui - lamda * P[u, :])
                Q[:, i] = Q[:, i] + alpha * (P[u, :] * eui - lamda * Q[:, i])



        alpha *= 0.9  # 每次迭代降低学习率
        # # 计算损失函数值
        cost = 0
        # for u in range(M):
        #     for i in range(N):
        #         if Sample[user_list[u]].get(item_list[i], -1) != -1:
        #             cost += (R[u, i] - 1 / (1 + np.exp(-np.dot(P[u, :], Q[:, i])))) ** 2
    # for k in range(K): #计算正则项
    # cost += lamda*(P[u,k]**2 + Q[k,i]**2)

    # if cost < 0.1*:
    # break

    R_new = pd.DataFrame(1 / (1 + np.exp(-np.dot(P, Q))), index=np.arange(M), columns=np.arange(N))
    return R_new, cost


def LFM_Recommand(user, R, N):
    '''

    :param user: 用户id
    :param R: 训练好的用户物品评分矩阵
    :param N: 推荐物品长度
    :return:
    推荐列表，字典形式。物品id:评分
    '''
    # NegativeSample = [i[0] for i in Sample[user].items() if i[1] == 0]

    ser = R.loc[user, :].sort_values(ascending=False)[:N]

    Recommend = dict(zip(ser.index, ser))  # 物品id:兴趣度

    return list(Recommend.keys())





