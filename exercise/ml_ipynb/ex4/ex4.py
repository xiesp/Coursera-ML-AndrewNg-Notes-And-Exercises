#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir('C:\\Users\\Administrator\\Desktop\\ml_ipynb\\ex4')
	print(os.getcwd())
except:
    print(os.getcwd()+" not changed!!!")


#%% [markdown]
# # ex4_nn  - 神经网络


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
import scipy.optimize as opt
import sys
sys.path.append('..')
sys.path
from helper import formulas




#%% [markdown]
# ## 1 加载数据

#%%
data = loadmat('ex4data1.mat')
#注意data是一个dict,键是X和y.
print(data)
X = data['X']
y = data['y'].ravel()
print(X.shape)
print(y.shape)



#%% [markdown]
# ## 2 测试一下代价函数

#%% [markdown]
# 首先加载训练好的额参数


#%%
input_layer_size  = 400
m = X.shape[0]
hidden_layer_size = 25
num_labels = 10
# 加载参数
parameters = loadmat('ex4weights.mat')
loaded_Theta1 = parameters['Theta1']
loaded_Theta2 = parameters['Theta2']
print("Theta1 and Theta2 shape is{} and {}".format(loaded_Theta1.shape,
    loaded_Theta2.shape))


#%% [markdown]
# 处理一下参数



#%%
# 处理一下对比的Y矩阵
Y = np.zeros((m,num_labels))
# 特别注意,数据是matlab生成的,matlab是以1为开始下标的,
# 但是py是0,而且y的数据当中,以10代表0
for i in range(m):
    # label值是1-10
    label = y[i]
    # 所以我们-1,变成0-9
    label -= 1
    # 这样Y矩阵的每一行,0-9下标的数据,存储的是1-9,10数字,和matlab代码等价
    # 这样,下面测试代价函数才能正确
    Y[i,label] = 1

# 把加载的参数变成矩阵
load_params = formulas.reshapeThetaToVector(loaded_Theta1,loaded_Theta2,
    input_layer_size,hidden_layer_size,num_labels)



#%% [markdown]
# 开始测试已经训练过的参数,看看代价函数是否正确
# 首先不包括正则化部分,lmd = 0


#%%
lmd = 0
# 计算代价函数
J = formulas.nnCost(load_params,X,Y,lmd,input_layer_size,hidden_layer_size,num_labels)
print("With lmd = 0,Cost at parameters is {},this value should be 0.287629".format(J))


#%% [markdown]
# 再测试包括正则化部分,lmd = 1

#%%
lmd = 1
# 计算代价函数
J = formulas.nnCost(load_params,X,Y,lmd,input_layer_size,hidden_layer_size,num_labels)
print("With lmd = 1,Cost at parameters is {},this value should be 0.383770".format(J))


#%% [markdown]
# OK 经过上一步,我们已经知道,向前传播和代价函数都是没问题的
# 现在我们开始反向传播计算提图,一遍训练算法





#%% [markdown]
# ## 3 开始训练算法
# 首先,需要随机初始化参数

#%%
# 随机初始化函数
def randomIntializeWeight(inputSize,outSize):
    W = np.zeros((outSize,inputSize+1))
    eps = 0.12
    W = np.random.random((outSize,inputSize+1)) * 2 * eps - eps
    return W


#%%
# 初始化参数
init_Theta1 = randomIntializeWeight(input_layer_size,hidden_layer_size)
init_Theta2 = randomIntializeWeight(hidden_layer_size,num_labels)
# 合并成一整个向量
nn_params = formulas.reshapeThetaToVector(init_Theta1,init_Theta2,
    input_layer_size,hidden_layer_size,num_labels)


#%% [markdown]
# 然后,开始训练

#%%

# 额,下面的的慢的根本没法运行,太慢了,半个钟都没出结果
# result = opt.minimize(fun=formulas.nnCost,
#         x0=nn_params, jac=formulas.backPropagate, 
#         args=(X, Y, lmd,input_layer_size,hidden_layer_size,num_labels),
#         options={'maxiter': 400})

# 这个算法很快,但是准确度怎么只有85%?
# result = opt.fmin_tnc(func=formulas.nnCost,
#         x0=nn_params, fprime=formulas.backPropagate, 
#         args=(X, Y, lmd,input_layer_size,hidden_layer_size,num_labels))


# fmin_cg又快,准确度也好高,99%!!!!
result = opt.fmin_cg(f=formulas.nnCost,
        x0=nn_params, fprime=formulas.backPropagate, 
        maxiter=400,disp=True,full_output=True,
        args=(X, Y, lmd,input_layer_size,hidden_layer_size,num_labels))


#%%
# 取出最后的参数,result[1]是循环的次数,循环了573次...
opt_theta_vec = result[0]
# 我们需要恢复成矩阵形式
[opt_Theta1,opt_Theta2] = formulas.reshapeToThetaMatrix(opt_theta_vec,
        input_layer_size,hidden_layer_size,num_labels)


#%% 
# 看看这时候代价函数的值是多少
J = formulas.nnCost(opt_theta_vec,X,Y,lmd,
input_layer_size,hidden_layer_size,num_labels)
print("With lmd = 1,Cost after train is {}".format(J))




#%% [markdown]
# ## 4看一下准确度

#%%
Y_prob = formulas.forwardPropagate(X,opt_Theta1,opt_Theta2)[4]
# 记住,下标需要+1
y_pred = np.argmax(Y_prob,axis= 1) +1
print('Train accuracy: {}'.format(np.mean(y == y_pred) * 100))

#%%
