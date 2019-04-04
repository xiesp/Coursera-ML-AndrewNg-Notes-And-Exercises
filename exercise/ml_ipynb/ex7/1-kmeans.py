#%%
import os
try:
	os.chdir('C:\\Users\\Administrator\\Desktop\\ml_ipynb\\ex7')
	print(os.getcwd())
except:
    print(os.getcwd()+" not changed!!!")

#%%
%matplotlib inline
%reload_ext autoreload
%autoreload 2
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from helper import kmeans as km


#%% [markdown]
# # kmeans算法实现

#%% [markdown]
# ## 1 加载数据和查看数据

#%%
mat = sio.loadmat('./data/ex7data2.mat')
data2 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
print(data2.head())

sns.set(context="notebook", style="white")
sns.lmplot('X1', 'X2', data=data2, fit_reg=False)


#%% [markdown]
# ## 2 随机初始化
# 随机选择几个中心点,然后画出图像

#%%
init_centroids  = km.random_init(data2, 3)


#%%
x = np.array([1, 1])
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(x=init_centroids[:, 0], y=init_centroids[:, 1])

for i, node in enumerate(init_centroids):
    ax.annotate('{}: ({},{})'.format(i, node[0], node[1]), node)
ax.scatter(data2['X1'], data2['X2'], marker='.', s=10)


#%% [markdown]
# ## 3 运行一次中心聚类,看看数据
# 并且画出图像看看


#%%
# 寻找下标
C = km.assign_cluster(data2, init_centroids)
data_with_c = km.combine_data_C(data2, C)
data_with_c.head()

#%%
sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)


#%% [markdown]
# ## 4 寻找新的中心点
# 并且看看画出来的效果


#%%
# 根据当前下标寻找中心点,这个方法运行多次就可以看出一次次的变化
new_centorids = km.new_centroids(data2, C)
# 形成新的下标
C = km.assign_cluster(data2, new_centorids)
data_with_c = km.combine_data_C(data2, C)
#data_with_c.head()
sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)



#%% [markddown]
# ## 5 整个流程一起实现

#%%
# 注意,这里有可能效果不好,多运行几次把
final_C, final_centroid, _= km._k_means_iter(data2, 3)
data_with_c = km.combine_data_C(data2, final_C)
sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)
print(km.cost(data2, final_centroid, final_C))

#%% [markdown]
# ## 6多次随机初始化
# 经过第5步,我们知道,最后运行出来的不一定是最优的
# 所以我们可以把多次随机初始化和运行放到一起,一次性解决


#%%
# 可以看到,效果不错
best_C, best_centroids, least_cost = km.k_means(data2, 3)
print(least_cost)
data_with_c = km.combine_data_C(data2, best_C)
sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)



#%% [markdown]
# ## 7 尝试使用sklearn

#%%
from sklearn.cluster import KMeans


#%%
sk_kmeans = KMeans(n_clusters=3)
sk_kmeans.fit(data2)


#%%
sk_C = sk_kmeans.predict(data2)
data_with_c = km.combine_data_C(data2, sk_C)
sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)

#%%
