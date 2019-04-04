import numpy as np
from sigmoid import *

def predict(theta1, theta2, x):
    # Useful values
    m = x.shape[0]
    num_labels = theta2.shape[0]

    # You need to return the following variable correctly
    p = np.zeros(m)

    # ===================== Your Code Here =====================
    # Instructions : Complete the following code to make predictions using
    #                your learned neural network. You should set p to a
    #                1-D array containing labels between 1 to num_labels.
    #
    x = np.c_[np.ones(m), x]
    tmp = sigmoid( np.matmul(x,theta1.T) )
    tmp = np.c_[np.ones(np.size(tmp,axis=0)), tmp]
    tmp = sigmoid( np.matmul(tmp,theta2.T) )
    p = np.argmax(tmp,axis=1)
    # py index start from 0
    p+=1
    return p


