def Recall_Precison(preds, test):
    '''
    :param train: 字典形式，测试集上给用户推荐的行为列表
    :param test: 字典形式，测试集上用户的行为列表
    :return:返回精准率和召回率
    '''
    hit = 0
    Ru_total = 0
    Tu_total = 0
    for user in preds.keys():
        Tu = set(test[user])
        Ru = set(preds[user])
        hit = hit + len(Ru & Tu)
        Ru_total = Ru_total + len(Ru)
        Tu_total = Tu_total + len(Tu)

    recall = hit / Tu_total
    precision = hit / Ru_total

    return precision, recall


def Coverage(train, I):
    '''
    :param train: 字典形式，训练集上给用户推荐的物品列表
    :param I: 测试集上所有的物品个数
    :return:返回覆盖率
    '''
    hit = 0
    total = 0
    R = set()  # 推荐系统能够推荐出来的物品集合
    for user in train.keys():
        Ru = set(train[user])
        R = R | Ru

    return len(R) / I


def Popularity(train, Popularity_Di):
    '''
    :param train: 字典形式，训练集上给用户推荐的物品列表
    :param Popularity_Di: 字典形式，物品流行度
    :return:返回推荐列表的物品平均流行度
    '''
    R = set()  # 推荐系统能够推荐出来的物品集合
    for user in train.keys():
        Ru = set(train[user])
        R = R | Ru

    P = {key: Popularity_Di[key] for key in Popularity_Di if key in R}
    return sum(P.values()) / len(R)
