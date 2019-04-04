import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.figure()

    # ===================== Your Code Here =====================
    # Instructions : Plot the positive and negative examples on a
    #                2D plot, using the marker="+" for the positive
    #                examples and marker="o" for the negative examples
    #

    # or np.where(y==1)
    '''
    pos = (y==1)
    neg = (y==0)
    plt.scatter(X[pos,0],X[pos,1],marker='x')
    plt.scatter(X[neg,0],X[neg,1],marker='o')
    '''
    pos = X[np.where(y == 1)]
    neg = X[np.where(y == 0)]
    fig, ax = plt.subplots()
    ax.plot(pos[:, 0], pos[:, 1], "k+", neg[:, 0], neg[:, 1], "yo")
