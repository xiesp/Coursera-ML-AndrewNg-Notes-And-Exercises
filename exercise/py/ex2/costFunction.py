import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #

    # theta.shape = [n+1 1],x.shape[m n+1]+



    z = X @ theta
    g = sigmoid(z)

    # note:log(g) for y =1 and log(1-g) for y = 0
    # so where g is 0's or 1's,we got NAN for the cost
    # but the np compliant for "divided by zero..."
    # why trigger this?or this is a bug in numpy?
    # so the things wo could do is using the numerical trick of
    # "add epsilon of very small data"
    eps = 1e-5
    cost = -1.0 / m * (y.T @ np.log(g+eps) + (1-y).T @ np.log(1-g+eps))
    grad = 1.0 / m * (X.T @ (g-y))

    # ===========================================================

    return cost, grad
