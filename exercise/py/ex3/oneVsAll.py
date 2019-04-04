import scipy.optimize as opt
import lrCostFunction as lCF
from sigmoid import *


def one_vs_all(X, y, num_labels, lmd):
    # Some useful variables
    (m, n) = X.shape

    # You need to return the following variables correctly
    all_theta = np.array([])  # initialize theta

    # Add ones to the X data 2D-array
    X = np.c_[np.ones(m), X]


    for i in range(num_labels):
        print('Optimizing for handwritten number {}...'.format(i))
        # ===================== Your Code Here =====================
        # Instructions : You should complete the following code to train num_labels
        #                logistic regression classifiers with regularization
        #                parameter lambda
        #
        #
        # Hint: you can use y == c to obtain a vector of True(1)'s and False(0)'s that tell you
        #       whether the ground truth is true/false for this class
        #
        # Note: For this assignment, we recommend using opt.fmin_cg to optimize the cost
        #       function. It is okay to use a for-loop (for c in range(num_labels) to
        #       loop over the different classes

        initial_theta = np.zeros(n+1)
        # initialize y values
        yTemp = np.zeros(y.shape[0])

        # !!!!Note that the the y's are 1-10,not 0-9!!
        yTemp[np.where(y == (i+1))] = 1
        result = opt.minimize(fun=lCF.lr_cost_function,x0=initial_theta,args=(X,yTemp,lmd),
                     method="TNC",jac=lCF.lr_grad_function)

        all_theta =np.append(all_theta, result.x)
        # ============================================================
        print('Done')

    # Struggled on this for awhile.
    # Reshape works from left to right, top to bottom.
    # So if your data needs to be in columns instead of rows,
    # It messes it all up, but it still "works"
    all_theta = np.reshape(all_theta, (num_labels, n + 1))
    return all_theta
