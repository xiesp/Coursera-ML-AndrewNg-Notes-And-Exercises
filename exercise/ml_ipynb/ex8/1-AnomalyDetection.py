#%%
import os
try:
	os.chdir('C:\\Users\\Administrator\\Desktop\\ml_ipynb\\ex8')
	print(os.getcwd())
except:
    print(os.getcwd()+" not changed!!!")



#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white", 
    palette=sns.color_palette("RdBu"),color_codes= False)
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
import sys
sys.path.append('..')
from helper import anomaly
from sklearn.model_selection import train_test_split

#%% [markdown]
# # 异常检测


#%% [markdown]
# ## 1 加载数据和展示
# 数据集里面有训练集和验证集
# 没有测试集

#%%
mat = sio.loadmat('./data/ex8data1.mat')
print(mat.keys())
# 训练集,307个
X = mat.get('X')
# 验证集,307个
Xval = mat.get('Xval')
yval = mat.get('yval')


#%%
# 把验证集一般划分为测试集
Xval, Xtest, yval, ytest = train_test_split(mat.get('Xval'),
            mat.get('yval').ravel(),
            test_size=0.5)


#%%
# 展示训练集数据
sns.regplot('Latency', 'Throughput',
           data=pd.DataFrame(X, columns=['Latency', 'Throughput']), 
           fit_reg=False,
           scatter_kws={"s":20,
                        "alpha":0.5})


#%% [markdown]
# ## 2 看看数据的状态
# 注意,看协方差矩阵,就知道数据有没有相关性了

#%%
mu = X.mean(axis=0)
print(mu, '\n')
cov = np.cov(X.T)
print(cov)

#%%
# example of creating 2d grid to calculate probability density
np.dstack(np.mgrid[0:3,0:3])

#%%
# create multi-var Gaussian model
multi_normal = stats.multivariate_normal(mu, cov)

# create a grid
x, y = np.mgrid[0:30:0.01, 0:30:0.01]
pos = np.dstack((x, y))

fig, ax = plt.subplots()

# plot probability density
ax.contourf(x, y, multi_normal.pdf(pos), cmap='Blues')

# plot original data points
sns.regplot('Latency', 'Throughput',
           data=pd.DataFrame(X, columns=['Latency', 'Throughput']), 
           fit_reg=False,
           ax=ax,
           scatter_kws={"s":10,
                        "alpha":0.4})

#%% [markdown]
# ## 3 选择阈值
# 使用训练基训练出模型,验证集选择最好的eps(F1)
# 最后测试集评估

#%%
def select_threshold(X, Xval, yval):
    """use CV data to find the best epsilon
    Returns:
        e: best epsilon with the highest f-score
        f-score: such best f-score
    """
    # create multivariate model using training data
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    multi_normal = stats.multivariate_normal(mu, cov)

    # this is key, use CV data for fine tuning hyper parameters
    pval = multi_normal.pdf(Xval)

    # set up epsilon candidates
    epsilon = np.linspace(np.min(pval), np.max(pval), num=10000)

    # calculate f-score
    fs = []
    for e in epsilon:
        y_pred = (pval <= e).astype('int')
        fs.append(f1_score(yval, y_pred))

    # find the best f-score
    argmax_fs = np.argmax(fs)

    return epsilon[argmax_fs], fs[argmax_fs]


