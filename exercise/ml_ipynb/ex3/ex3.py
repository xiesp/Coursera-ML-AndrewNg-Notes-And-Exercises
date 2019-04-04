#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir('C:\\Users\\Administrator\\Desktop\\ml_ipynb\\ex3')
	print(os.getcwd())
except:
    print(os.getcwd()+" not changed!!!")


#%% [markdown]
# # ex3  - 多元分类和神经网络入门


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
import sys
sys.path.append('..')
sys.path
from helper import formulas





#%% [markdown]
# ## 1 多元分类
# 使用逻辑回归实现多元分类,本质上就是建立多个二元逻辑回归 

#%% [markdown]
# ### 1.1 加载数据

#%%
data = loadmat('ex3data1.mat')
#注意data是一个dict,键是X和y.
print(data)
#看看前面几行数据
X = data['X']
y = data['y'].ravel()
print(X.shape)
print(y.shape)




#%% [markdown]
# ### 1.2 运行算法
# 图像在martix X中表示为400维向量（其中有5,000个）。 400维“特征”是原始20 x 20图像中每个像素的灰度强度。
# 需要注意,y标签是1-10,0用10表示!!!!
# 代价函数和梯度已经在formulas写好了,复用ex2的即可,下面是实现算法的流程


#%%
# 首先初始化一些有用的变量
[m,n] = X.shape
print("m and n is:"+ str(m)+","+str(n))
# 一共有多少分类
num_lambels = 10
all_theta = np.zeros((num_lambels,n+1))
#为X增加一列1
X= np.c_[np.zeros((m,1)),X]
lmd = 0.1
# 把y标签的10改为0,方便处理
y[np.where(y == 10)] = 0


#%%
#开始循环,注意这里i是0-9
for i in range(10):
    initial_theta = np.zeros(n+1)
    # i+1是因为i从0开始计算
    # 这里每一次的y标签都是不同的
    # 注意需要ravel
    cur_label = np.where(y ==i,1,0).ravel()
    # 使用系统自带函数
    cur_opt_result = opt.fmin_tnc(func=formulas.logisticReg_cost_reg,
        x0=initial_theta, fprime=formulas.logisticReg_grad_reg, args=(X, cur_label, lmd))
    #print(cur_opt_result[0].shape)
    # 保存下本次的参数
    all_theta[i,:] = cur_opt_result[0]





#%% [markdown]
# ### 1.3 得到算法的准确度


#%%
# 每一个样本都有10个预测结果,概率只取最大的那个
y_prob = formulas.sigmoid(X @ all_theta.T)
y_pred =np.argmax(y_prob,axis= 1)
# 精确度95.88,比用10表示0提高了差不多1%.
# 标准答案的确定度应该是94.96
print('Train accuracy: {}'.format(np.mean(y == y_pred) * 100))


#%%
