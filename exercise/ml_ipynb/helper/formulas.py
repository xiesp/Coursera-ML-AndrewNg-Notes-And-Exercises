import numpy as np


"""
线性回归代价函数
参数:
    X矩阵,X.shape=[m,n]
    y列向量,y.shape = [m,1]
    theta列向量,theta.shape = [n,1]
"""
def linearReg_cost(X,y,theta):
    inner = np.power(((X @ theta) - y), 2)
    return np.sum(inner) / (2 * len(X))



"""
正则化线性回归代价函数
"""
def linearReg_cost_reg(theta,X,y,lmd):
    J = linearReg_cost(X,y,theta)
    # 增加正则化部分
    J += lmd / (2 * len(X)) * np.sum(theta[1:]**2)
    return J


"""
正则化线性回归梯度
"""
def linearReg_grad_reg(theta,X, y, lmd):
    m = X.shape[0]
    grad = X.T @ ((X @ theta) - y) / m
    # 加上正则化部分
    grad[1:] += theta[1:] * (lmd / m)
    return grad

    


"""
线性回归梯度下降
参数:
    X矩阵,X.shape=[m,n]
    y列向量,y.shape = [m,1]
    theta列向量,theta.shape = [n,1]
    alpha:learning rate
    iters:循环次数
"""
def linearReg_gradDesc(X, y, theta, alpha, iters):
    m = X.shape[0]
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = X.T @ ((X @ theta) - y)
        theta = theta - alpha / m * error
        cost[i] = linearReg_cost(X, y, theta)
    return theta, cost










"""
正规方程求解
    X矩阵,X.shape=[m,n]
    y列向量,y.shape = [m,1]
"""
def normalEqn(X, y):
    theta = np.linalg.pinv(X.T@X)@X.T@y
    return theta



"""
Sigmoid函数
"""
def sigmoid(z):
    return 1 / (1 + np.exp(-z))




"""
逻辑回归代价函数
假设参数已经全部是矩阵

---参数---
    X矩阵,X.shape=[m,n]
    y列向量,y.shape = [m,]
    theta列向量,theta.shape = [n,]
"""
def logisticReg_cost(theta,X, y):
    # 样本数目
    m = X.shape[0]
    z = X @ theta  
    g = sigmoid(z)
    # g.shape=[m,1],y.T.shape = [1,m]
    cost = -1.0 /m * ( y.T @ np.log(g) + (1-y).T @ np.log(1-g) )
    return cost

"""
逻辑回归梯度,假设参数已经全部是矩阵
注意,不需要循环多次,也不需要我们自己执行梯度下降,因为
我们将使用自带的优化函数.所以只要返回梯度就可以了

---参数---
    X矩阵,X.shape=[m,n]
    y列向量,y.shape = [m,]
    theta列向量,theta.shape = [n,]

---返回值---
当前参数的梯度

"""
def logisticReg_grad(theta,X, y): 
    z = X @ theta
    m = X.shape[0]
    g = sigmoid(z)
    # 初始化正确的大小
    #grad = np.zeros(theta.shape)
    # 计算梯度,X.T.shape = [n m],(g-y).shape = [m 1]
    grad = 1.0 /m * ( X.T @ ( g- y ))
    return grad.ravel()







"""
维度隐射
    degree:需要隐射多少维

"""
def mapFeatures(x1,x2,degree):
    # first column is all 1's
    result = np.ones(x1.shape[0])
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            result = np.c_[result, (x1**(i-j)) * (x2**j)]
    return result



    
 





"""
正则化逻辑回归代价函数
---参数---
    X矩阵,X.shape=[m,n]
    y列向量,y.shape = [m,]
    theta列向量,theta.shape = [n,]
"""
def logisticReg_cost_reg(theta,X, y,lmd):
    # 样本数目
    m = X.shape[0]
    z = X @ theta  
    g = sigmoid(z)
    # g.shape=[m,1],y.T.shape = [1,m]
    cost = -1.0 /m * ( y.T @ np.log(g) + (1-y).T @ np.log(1-g) ) + lmd /(2*m) * np.sum(theta[1:]**2)
    return cost






"""
正则化逻辑回归梯度

---参数---
    X矩阵,X.shape=[m,n]
    y列向量,y.shape = [m,]
    theta列向量,theta.shape = [n,]

"""
def logisticReg_grad_reg(theta,X, y,lmd): 
    z = X @ theta
    m = X.shape[0]
    g = sigmoid(z)
    # 初始化正确的大小
    #grad = np.zeros(theta.shape)
    # 计算梯度,X.T.shape = [n m],(g-y).shape = [m 1]
    grad = 1.0 /m * ( X.T @ ( g- y ))
    grad[1:] += lmd/m * theta[1:]
    return grad.ravel()






"""
计算sigmoid函数的导数
"""

def sigmoidGradient(z):
    g = sigmoid(z)
    # py当中 * 就表示矩阵的点乘
    g = g  * (1-g)
    return g




"""
将thtea向量恢复成矩阵
"""
def reshapeToThetaMatrix(nn_params,in_size,hidden_size,out_size):
    theta1end = (in_size+1) * hidden_size
    Theta1 = np.reshape(nn_params[0:theta1end],(hidden_size,in_size+1))
    Theta2 = np.reshape(nn_params[theta1end:nn_params.size],(out_size,hidden_size+1))
    return Theta1,Theta2



"""
展开Theta矩阵变成向量
"""
def reshapeThetaToVector(Theta1,Theta2,in_size,hidden_size,out_size):
    vec1 = Theta1.reshape((in_size+1) * hidden_size,1).ravel()   
    vec2 = Theta2.reshape((hidden_size+1) * out_size,1).ravel()
    # 合并成单一一个向量
    nn_params = np.hstack((vec1,vec2))
    return nn_params




"""
神经网络向前传播算法
"""
def forwardPropagate(X, theta1, theta2):
    m = X.shape[0]
    # 插入一列1
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    # a2再插入一列
    a2 = np.insert(a2, 0, values=np.ones(m), axis=1)
    z3 = a2 @ theta2.T
    Y_pred = sigmoid(z3)
    
    return a1, z2, a2, z3, Y_pred


"""
神经网络代价函数
除了lmd是正则化参数,其他都是矩阵
"""
def nnCost(nn_params,X,Y,lmd,in_size,hidden_size,out_size):
    # 恢复Theta矩阵
    [Theta1,Theta2] = reshapeToThetaMatrix(nn_params,in_size,hidden_size,out_size)
    # 计算向前传播的值
    [a1, z2, a2, z3, Y_pred] = forwardPropagate(X,Theta1,Theta2)

    [m,n] = Y.shape
    # 注意py当中,*相当于点乘
    J = Y * np.log(Y_pred) + (1-Y) * np.log(1-Y_pred)
    J = -1/m * np.sum(J)
    # 然后是正则化的部分,注意第一列不参与正则化
    theta1Square = Theta1[:,1:]**2
    theta2Square = Theta2[:,1:]**2
    J += lmd /(2.0 * m) * ( np.sum(theta1Square) + np.sum(theta2Square) )
    return J



"""
神经网络反向传播算法实现
"""
def backPropagate(nn_params,X,Y,lmd,in_size,hidden_size,out_size):
    # 恢复Theta矩阵
    [Theta1,Theta2] = reshapeToThetaMatrix(nn_params,in_size,hidden_size,out_size)
    # 计算向前传播的值
    [a1, z2, a2, z3, Y_pred] = forwardPropagate(X,Theta1,Theta2)

    m = Y.shape[0]
    delta3 = Y_pred - Y
    # 注意,乘法还是点乘的意思
    delta2 = ( Theta2[:,1:].T @ delta3.T ).T * sigmoidGradient(z2)
    # 计算梯度矩阵
    Delta2 = delta3.T @ a2
    Delta1 = delta2.T @ a1
    #
    Theta2_grad =  Delta2 / m
    Theta1_grad =  Delta1 / m
    # 开始正则化处理,不应该处理bias term
    Theta1[:,1] = 0
    Theta2[:,1] = 0
    Theta1_grad = Theta1_grad + lmd/m * Theta1
    Theta2_grad = Theta2_grad + lmd/m * Theta2
    # 转换成向量返回
    grad = reshapeThetaToVector(Theta1_grad,Theta2_grad,
        in_size,hidden_size,
        out_size)
    return grad