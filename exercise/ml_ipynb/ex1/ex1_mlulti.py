#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'ex1'))
    print(os.getcwd())
except:
	pass

#%% [markdown]
# # ex1_multi - 多变量线性回归


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#引入公式包
import sys
sys.path.append('..')
from helper import formulas



#%%
path =  'ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data.describe()

#%% [markdown]
# ## 特征归一化。

#%%
data = (data - data.mean()) / data.std()
data.head()

#%% [markdown]
# ## 开始运行线性回归

#%%
# add ones column
alpha = 0.01
iters = 1000
data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
# 转换为np对象,初始化theta
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.zeros([3,1]))
# 开始梯度下降
thetaOpt, cost = formulas.linearReg_gradDesc(X, y, theta, alpha, iters)
# 得到梯度下降后的代价
formulas.linearReg_cost(X, y, thetaOpt)

#%% [markdown]
# ## 看一下训练过程

#%%
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()



#%% [markdown]
# ## 正规方程
# 可以看到正规方程的cost小了一点点,是真正的最优解!!!

#%%
thetaOpt2=formulas.normalEqn(X, y)
costLinear = formulas.linearReg_cost(X,y,thetaOpt)
costNormal = formulas.linearReg_cost(X,y,thetaOpt2)
costLinear,costNormal



#%% [markdown] 
# ## 看看不同的学习率之间的区别



#%%
alphaArr = np.logspace(-1, -5, num=4)
# 这样处理过后就是3倍的区间了
candidate = np.sort(np.concatenate((alphaArr, alphaArr*3)))
print(candidate)


#%%
fig, ax = plt.subplots(figsize=(16, 9))
for alpha in candidate:
    _, cost_data = formulas.linearReg_gradDesc(X,y,theta,alpha,iters)
    ax.plot(np.arange(iters), cost_data, label=alpha)

ax.set_xlabel('iterations', fontsize=18)
ax.set_ylabel('cost', fontsize=18)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('learning rate', fontsize=18)
plt.show()

