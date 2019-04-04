import numpy as np
from sigmoid import *


def lr_cost_function(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #
    eps = 0
    z = np.matmul(X,theta)
    g = sigmoid(z)
    cost = -1.0 / m * (y.T @ np.log(g+eps) + (1-y).T @ np.log(1-g+eps)) + \
             lmd/(2*m) * np.sum(theta[1:]**2)
    return cost



def lr_grad_function(theta, X, y, lmd):
    m = y.size
    grad = np.zeros(theta.shape)
    z = X @ theta
    g = sigmoid(z)
    grad = 1.0 / m * (X.T @ (g-y))
    grad[1:] += lmd / m * theta[1:]
    return grad
