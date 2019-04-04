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
sns.set(context="notebook", style="white")

import numpy as np
import pandas as pd
import scipy.io as sio

import sys
sys.path.append('..')

from helper import general
from helper import pca

#%% [markdown]
# # 使用2维数据进行PCA

#%% [markdown]
# ## 1 加载数据和展示

#%%
mat = sio.loadmat('./data/ex7data1.mat')
X = mat.get('X')
# visualize raw data
print(X.shape)
sns.lmplot('X1', 'X2', 
           data=pd.DataFrame(X, columns=['X1', 'X2']),
           fit_reg=False)

#%% [markdown]
# ## 2 开始运行算法
# 首先均值归一
# 可以看到,数据基本在-1,1之间


#%%
X_norm = pca.normalize(X)
sns.lmplot('X1', 'X2', 
           data=pd.DataFrame(X_norm, columns=['X1', 'X2']),
           fit_reg=False)


#%% [markdown]
# 协方差矩阵

#%%
Sigma = pca.covariance_matrix(X_norm)  # capital greek Sigma
Sigma  # (n, n)


#%% [markdown]
#  开始运行PCA

#%%
U, S, V = pca.pca(X_norm)
Z = pca.project_data(X_norm, U, 1)
Z[:10]

#%% [markdown]
# ## 4 展示降维后的数据

#%%
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
sns.regplot('X1', 'X2', 
           data=pd.DataFrame(X_norm, columns=['X1', 'X2']),
           fit_reg=False,
           ax=ax1)
ax1.set_title('Original dimension')

sns.rugplot(Z, ax=ax2)
ax2.set_xlabel('Z')
ax2.set_title('Z dimension')



#%% [markdown]
# ## 5 看看恢复数据

#%%
X_recover = pca.recover_data(Z, U)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 4))

sns.rugplot(Z, ax=ax1)
ax1.set_title('Z dimension')
ax1.set_xlabel('Z')

sns.regplot('X1', 'X2', 
           data=pd.DataFrame(X_recover, columns=['X1', 'X2']),
           fit_reg=False,
           ax=ax2)
ax2.set_title("2D projection from Z")

sns.regplot('X1', 'X2', 
           data=pd.DataFrame(X_norm, columns=['X1', 'X2']),
           fit_reg=False,
           ax=ax3)
ax3.set_title('Original dimension')

#%%
