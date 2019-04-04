import numpy as np

# data shape (pd.DataFrame)
#          X1        X2
# 0  1.842080  4.607572
# 1  5.658583  4.799964
# 2  6.352579  3.290854
# 3  2.904017  4.612204
# 4  3.231979  4.939894


# support fn --------------------------------
def combine_data_C(data, C):
    data_with_c = data.copy()
    data_with_c['C'] = C
    return data_with_c


# 随机初始化函数,随机选择k个数据点
"""
Args:
    data: DataFrame
    k: int
Returns:
    k samples: ndarray
"""
def random_init(data, k):
    return data.sample(k).as_matrix()




# 找到所有样本的中心点下标
# 注意x表示单一一个样本
"""
Args:
    x: ndarray (n, ) -> n features
    centroids: ndarray (k, n)
Returns:
    k: int
"""
def _find_your_cluster(x, centroids):
    # apply_along_axis 可以使用一个函数对一个axis的所有数据进行操作
    # axis = 1表示对每一行进行操作
    # arr=centroids - x 表示所有中心点都和样本点x做一次减法
    distances = np.apply_along_axis(
        # this give you l2 norm
        func1d=np.linalg.norm,  
        axis=1,
        # use ndarray's broadcast
        arr=centroids - x)  
    # 最后面得到了所有中心点和x样本点的距离,返回最短距离的下标
    # 返回下标
    return np.argmin(distances)


# 找到每一个样本点的最近中心点的下标
def assign_cluster(data, centroids):

    return np.apply_along_axis(
        # 找到所有最小距离形成的 ndarray
        lambda x: _find_your_cluster(x, centroids),
        # 对每一行,也就是每个样本就行处理
        axis=1,
        # 要处理的每一行数据就是所有样本
        arr=data.as_matrix())



# 计算新的中心点,C是这当前所有中心点的下标
def new_centroids(data, C):
    data_with_c = combine_data_C(data, C)

    # 按照C的值先分组
    # 最后求每一组的平均值
    # 根据C排序
    # 再把C的值去掉
    return data_with_c.groupby('C', as_index=False).\
                       mean().\
                       sort_values(by='C').\
                       drop('C', axis=1).\
                       as_matrix()


# k-means的代价函数
# C是当前所有所有样本分配的中心点下标
def cost(data, centroids, C):
    m = data.shape[0]
    # 这里的意思是,形成和data一样大小的矩阵
    # 但是每一行都是对应样本的中心点
    # 所以可以使用下面的broadcast方法
    expand_C_with_centroids = centroids[C]

    distances = np.apply_along_axis(
        func1d=np.linalg.norm,
        axis=1,
        # 对每一行都处理
        arr=data.as_matrix() - expand_C_with_centroids)
    return distances.sum() / m


# 完成k-means算法实现
# 可以根据tol提前结束
def _k_means_iter(data, k, epoch=100, tol=0.0001):
    # 随机初始化
    centroids = random_init(data, k)
    cost_progress = []
    # 循环次数
    for i in range(epoch):
        print('running epoch {}'.format(i))
        # 建立下标数组
        C = assign_cluster(data, centroids)
        # 根据下标创建新中心
        centroids = new_centroids(data, C)
        cost_progress.append(cost(data, centroids, C))
        # 误差小于当前的0.1%就提前退出
        if len(cost_progress) > 1:  # early break
            if (np.abs(cost_progress[-1] - cost_progress[-2])) / cost_progress[-1] < tol:
                break

    return C, centroids, cost_progress[-1]

# 多次实际参数,然后返回最好的结果
"""
Args:
    data (pd.DataFrame)
Returns:
    (C, centroids, least_cost)
"""
def k_means(data, k, epoch=100, n_init=10):

    # 连续运行n_init次,全部结果存成一个数组
    tries = np.array([_k_means_iter(data, k, epoch) for _ in range(n_init)])
    # 以最小代价的返回
    least_cost_idx = np.argmin(tries[:, -1])

    return tries[least_cost_idx]
