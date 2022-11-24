from numpy import *
from sklearn.metrics.pairwise import euclidean_distances
random.seed(1234)

# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))  # 求这两个矩阵的距离，vector1、2均为矩阵


# init centroids with random samples
# 在样本集中随机选取k个样本点作为初始质心
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape  # 矩阵的行数、列数
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples), )  # 随机产生一个浮点数，然后将其转化为int型
        centroids[i, :] = dataSet[index, :]
    return centroids


# k-means cluster
# dataSet为一个矩阵
# k为将dataSet矩阵中的样本分成k个类
def k_means(dataSet, k, max_clustering=300, privacybudget=None):
    numSamples = dataSet.shape[0]  # 读取矩阵dataSet的第一维度的长度,即获得有多少个样本数据
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = mat(zeros((numSamples, 2)))  # 得到一个N*2的零矩阵
    clusterChanged = True

    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)  # 在样本集中随机选取k个样本点作为初始质心
    if privacybudget != None:
        radius = []    #radius of data for global sensitivity calculations -差分隐私使用
        for i in range(dataSet.shape[1]):
            radius.append(((dataSet[:, i].max() + 1) - dataSet[:, i].min()) / 2)
        global_sen = array(radius)*dataSet.shape[1]+1
    iter_count_flag = 0
    while clusterChanged and iter_count_flag <= max_clustering:
        clusterChanged = False
        ## for each sample
        for i in range(numSamples):  # range
            minDist = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            # 计算每个样本点与质点之间的距离，将其归内到距离最小的那一簇
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

                    ## step 3: update its cluster
            # k个簇里面与第i个样本距离最小的的标号和距离保存在clusterAssment中
            # 若所有的样本不在变化，则退出while循环
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2  # 两个**表示的是minDist的平方

        if privacybudget != None:
            noise1 = []
            for i in range(dataSet.shape[1]):
                noise1.append(random.laplace(global_sen[i] / privacybudget))
            noise1 = array(noise1)
            noise2 = random.laplace(global_sen[i] / privacybudget)
        ## step 4: update centroids
        for j in range(k):
            # clusterAssment[:,0].A==j是找出矩阵clusterAssment中第一列元素中等于j的行的下标，返回的是一个以array的列表，第一个array为等于j的下标
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]  # 将dataSet矩阵中相对应的样本提取出来
            if pointsInCluster.shape[0] != 0:
                if privacybudget != None:
                    centroids[j, :] = (pointsInCluster.sum(0) + noise1)/ (pointsInCluster.shape[0] + noise2)# 计算增加差分隐私后的标注为j的所有样本的平均值
                else:
                    centroids[j, :] = mean(pointsInCluster, axis=0) # 计算标注为j的所有样本的平均值
        iter_count_flag += 1

    print ("actually iterate number is : {}".format(iter_count_flag))
    print('Congratulations, cluster complete!')
    return euclidean_distances(dataSet, centroids), clusterAssment
