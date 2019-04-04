import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #

    z = X @ theta
    g = sigmoid(z)
    eps = 0
    cost = -1.0 / m * (y.T @ np.log(g+eps) + (1-y).T @ np.log(1-g+eps)) + \
             lmd/(2*m) * np.sum(theta[1:]**2)
    grad = 1.0 / m * (X.T @ (g-y))
    grad[1:] += lmd / m * theta[1:]

    # ===========================================================

    return cost, grad
