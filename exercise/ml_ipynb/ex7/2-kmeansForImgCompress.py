#%%
import os
try:
	os.chdir('C:\\Users\\Administrator\\Desktop\\ml_ipynb\\ex7')
	print(os.getcwd())
except:
    print(os.getcwd()+" not changed!!!")


#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
import numpy as np
import pandas as pd

import sys
sys.path.append('..')

from helper import kmeans as km

#%% [markdown]
# # 使用k-means进行数据压缩



#%% [markdown]
# ## 1 首先加载数据看看

#%%
# http://scikit-image.org/
# cast to float, you need to do this 
# otherwise the color would be weird after clustring
pic = io.imread('data/bird_small.png')/255.
io.imshow(pic)
data = pic.reshape(128*128, 3)

#%% [markdown]
# ## 2 使用k-means进行数据压缩

#%%
# 使用自己写的果然特别慢啊啊.......4-5分钟
# 可以发现自己写的cpu使用只有30%,而sklearn可以到达100%,应该是没有利用多核
#C, centroids, cost = km.k_means(pd.DataFrame(data), 16, epoch = 10, n_init=3)
# 20秒左右,很快
from sklearn.cluster import KMeans
model = KMeans(n_clusters=16, n_init=100, n_jobs=-1)
model.fit(data)

#%%
centroids = model.cluster_centers_
print(centroids.shape)
C = model.predict(data)
print(C.shape)
# 这个就是近似数据
print(centroids[C].shape)


#%% [markdown]
# ## 3 开始恢复图像对比

#%%
compressed_pic = centroids[C].reshape((128,128,3))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)


#%%
