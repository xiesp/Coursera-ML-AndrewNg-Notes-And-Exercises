#%%
import os
try:
	os.chdir('C:\\Users\\Administrator\\Desktop\\ml_ipynb\\ex6')
	print(os.getcwd())
except:
    print(os.getcwd()+" not changed!!!")


#%%
from sklearn import svm
from sklearn import metrics
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.linear_model import LogisticRegression


#%% [markdown]
# # 区分垃圾邮件

#%% [markdown]
# ## 1 首先加载数据
# 有训练集和测试集2部分

#%%
mat_tr = sio.loadmat('data/spamTrain.mat')
mat_tr.keys()

#%%
X, y = mat_tr.get('X'), mat_tr.get('y').ravel()
X.shape, y.shape

#%%
mat_test = sio.loadmat('data/spamTest.mat')
mat_test.keys()


#%%
test_X, test_y = mat_test.get('Xtest'), mat_test.get('ytest').ravel()
test_X.shape, test_y.shape

#%% [markdown]
# ## 2 开始训练SVM模型


#%%
svc = svm.SVC()
svc.fit(X, y)



#%%
# 在测试集上看效果
pred = svc.predict(test_X)
print(metrics.classification_report(test_y, pred))



#%% [markdown]
# ## 3 看看逻辑回归怎么样
# 会发现逻辑回归表现的完全比SVM好.....


#%%
logit = LogisticRegression()
logit.fit(X, y)

#%%
pred = logit.predict(test_X)
print(metrics.classification_report(test_y, pred))


#%%
