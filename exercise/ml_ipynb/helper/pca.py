import numpy as np
import matplotlib.pyplot as plt

# X (ndarray)
# [[ 3.38156267  3.38911268]
#  [ 4.52787538  5.8541781 ]
#  [ 2.65568187  4.41199472]]


# support functions ---------------------------------------


# 把前n个数据画成图片
"""
n has to be a square number
"""
def plot_n_image(X, n):

    # 图片是正方形
    pic_size = int(np.sqrt(X.shape[1]))
    # 要展示多少张图片
    grid_size = int(np.sqrt(n))
    # 选出前n行数据
    first_n_images = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,
                sharey=True, sharex=True, figsize=(8, 8))

    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(
                first_n_images[grid_size * r + c].reshape(
                    (pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


# PCA functions ---------------------------------------



# 协方差矩阵
"""
Args:
    X (ndarray) (m, n)
Return:
    cov_mat (ndarray) (n, n):
        covariance matrix of X
"""
def covariance_matrix(X):
    m = X.shape[0]
    return (X.T @ X) / m


# 均值归一化
"""
    for each column, X-mean / std
"""
def normalize(X):
    X_copy = X.copy()
    m, n = X_copy.shape
    for col in range(n):
        X_copy[:, col] = (X_copy[:, col] - X_copy[:, col].mean()) / X_copy[:, col].std()

    return X_copy




#pca算法
"""
http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
Args:
    X ndarray(m, n)
Returns:
    U ndarray(n, n): principle components
"""
def pca(X):

    # 1. normalize data
    X_norm = normalize(X)

    # 2. calculate covariance matrix
    Sigma = covariance_matrix(X_norm)  # (n, n)

    # 3. do singular value decomposition
    # remeber, we feed cov matrix in SVD, since the cov matrix is symmetry
    # left sigular vector and right singular vector is the same, which means
    # U is V, so we could use either one to do dim reduction
    U, S, V = np.linalg.svd(Sigma)  # U: principle components (n, n)

    return U, S, V

# 使用特征向量投影数据
"""
Args:
    U (ndarray) (n, n)
Return:
    projected X (n dim) at k dim
"""
def project_data(X, U, k):

    m, n = X.shape
    if k > n:
        raise ValueError('k should be lower dimension of n')

    return X @ U[:, :k]


# 恢复数据
def recover_data(Z, U):
    m, n = Z.shape

    if n >= U.shape[0]:
        raise ValueError('Z dimension is >= U, you should recover from lower dimension to higher')

    return Z @ U[:, :n].T
