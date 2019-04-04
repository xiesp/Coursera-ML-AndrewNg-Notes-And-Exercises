#%%
import os
try:
	os.chdir('C:\\Users\\Administrator\\Desktop\\ml_ipynb\\ex6')
	print(os.getcwd())
except:
    print(os.getcwd()+" not changed!!!")


#%% [markdown]
# # ex6 尝试高斯内核函数


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as opt
import sys
from sklearn import svm
import scipy.io as sio
import seaborn as sns
sys.path.append('..')
sys.path
from helper import formulas



#%% [markdown]
# # 1 首先加载数据

#%%
mat = sio.loadmat('./data/ex6data2.mat')
print(mat.keys())
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')
data.head()

#%% [markdown]
# 看看一共有多少数据
#%%
data.shape

#%% [markdown]
# # 2 展示数据
# 可以很明显的看出来,这是一个非线性的

#%%
sns.set(context="notebook", style="white")
sns.lmplot('X1', 'X2', hue='y', data=data, 
           size=5, 
           fit_reg=False, 
           scatter_kws={"s": 10}
          )



#%% [markdown]
# # 3 首先使用内置的高斯核函数


#%%
svc = svm.SVC(C=100, kernel='rbf', gamma=10, probability=True)
svc

#%%
svc.fit(data[['X1', 'X2']], data['y'])
svc.score(data[['X1', 'X2']], data['y'])


#%% [markdown]
# the predict_proba will give you ndarray (data size, class)
# so if you just want to plot the decision contour of this binary example, choose one class and plot it


#%%
predict_prob = svc.predict_proba(data[['X1', 'X2']])[:, 0]

#%%
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'], data['X2'], s=30, c=predict_prob, cmap='Reds')