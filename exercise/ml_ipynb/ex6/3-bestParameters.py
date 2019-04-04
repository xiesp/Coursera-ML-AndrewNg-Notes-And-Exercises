#%%
import os
try:
	os.chdir('C:\\Users\\Administrator\\Desktop\\ml_ipynb\\ex6')
	print(os.getcwd())
except:
    print(os.getcwd()+" not changed!!!")


#%%
from sklearn import svm
#搜索最佳参数可以用这个实现,但是没有安装
#from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import numpy as np
import pandas as pd
import scipy.io as sio


#%% [markdown]
# # 寻找最佳的参数
# 对于第三个数据集，我们给出了训练和验证集，并且基于验证集性能为SVM模型找到
# 最优超参数。 虽然我们可以使用scikit-learn的内置网格搜索来做到这一点，
# 但是本着遵循练习的目的，我们将从头开始实现一个简单的网格搜索。

#%% [markdown]
# # 1 加载参数
# 可以看到有验证集



#%%
mat = sio.loadmat('./data/ex6data3.mat')
print(mat.keys())


#%%
# 把数据提取出来
training = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
training['y'] = mat.get('y')

cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])
cv['y'] = mat.get('yval')


#%%
print(training.shape)
training.head()

#%%
print(cv.shape)
cv.head()

#%%



#%% [markdwon]
# # 2 现在我们需要寻找最佳的参数C和高斯核函数当中的sigma参数
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC



#%%
# 手动设置一些参数
candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
combination = [(C, gamma) for C in candidate for gamma in candidate]
len(combination)

#%%
# 尝试所有的组合,得到最高的分数
search = []
for C, gamma in combination:
    svc = svm.SVC(C=C, gamma=gamma)
    svc.fit(training[['X1', 'X2']], training['y'])
    search.append(svc.score(cv[['X1', 'X2']],cv['y']))


#%%
# 选择出最高的分数
best_score = search[np.argmax(search)]
best_param = combination[np.argmax(search)]
print(best_score, best_param)

#%%
# 看看各种指标
best_svc = svm.SVC(C=100, gamma=0.3)
best_svc.fit(training[['X1', 'X2']], training['y'])
ypred = best_svc.predict(cv[['X1', 'X2']])
print(metrics.classification_report(cv['y'], ypred))

#%% [markdown]
# # 3 可以使用 sklearn GridSearchCV来选择最佳参数
# 但是现在没哟这个包,代码如下


#%%
# parameters = {'C': candidate, 'gamma': candidate}
# svc = svm.SVC()
# clf = GridSearchCV(svc, parameters, n_jobs=-1)
# clf.fit(training[['X1', 'X2']], training['y'])

#%%
# 这里也可以看出,选择的参数是一样的C=100,simga = 0.3
# clf.best_params_



#%%
# 看看多少分
# clf.best_score_

#%%
# 最终看看各种指标
# ypred = clf.predict(cv[['X1', 'X2']])
# print(metrics.classification_report(cv['y'], ypred))



# 会发现这个指标其实和自己选择的有点不同,为什么呢
# 下面是原来作者的想法
""" 
uriouly... they are not the same result. What?
It turns out that GridSearch will appropriate part of data 
as CV and use it to find the best candidate.
So the reason for different result is just that GridSearch 
here is just using part of training data to train because it need part of data as cv set
 """