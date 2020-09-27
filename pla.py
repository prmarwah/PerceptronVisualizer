import random
from sklearn.datasets import make_blobs, make_classification
from matplotlib import pyplot as plt
import numpy as np


def getPred(x, w):
    '''Returns predicted label for the data point x based on w'''
    x = np.reshape(x, [1, x.shape[0]])
    return np.matmul(x, np.transpose(w))[0] > 0


def getError(X, w, y):
    '''Returns error (number of misclassified samples) and an array of the indexes of misclassified samples'''
    preds = np.array([getPred(x, w)[0] for x in X])
    errors = preds != y
    return (sum(errors), np.where(errors)[0])


def plotPLA(X, w, y, pause = None):
    '''Plots the decision boundary given by w over the labeled data set.
       If pause is provided a non blocking plot is generated that will be displayed for pause seconds'''
    # plot decision boundary from w
    x1list = np.linspace(-5, 5, 1000) # Create 1-D arrays for x1,x2 dimensions
    x2list = np.linspace(-5, 5, 1000) 
    x1,x2 = np.meshgrid(x1list, x2list) # Create 2-D grid x1list,x2list values
    Z = x1*w[0][1] + x2*w[0][2] + w[0][0] # equation of line
    plt.contour(x1, x2, Z, levels=[0])

    # plot labeled data points
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r_')
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'b+')

    if(pause):
        plt.show(block=False)
        plt.pause(pause)
        plt.close()
    else:
        plt.show()


# generate 2d classification dataset - not always linearly separable
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_clusters_per_class=1, class_sep=2, flip_y=0)
n = X.shape[0] # save number of samples
plotX = X # normal X matrix for plots
X = np.array([np.append(1, x) for x in X]) # augmented X matrix

w = np.zeros([1,3]) # initialize w to zero vector

# variables for pocket PLA
w_best = w
error_best = n

plotPLA(plotX, w, y)

# perform pla
_iter = 0
max_iter = 50
error, misclassifieds = getError(X, w, y)
while(error > 0 and _iter < max_iter):
    # randomly choose a misclassified point
    i = random.choice(misclassifieds)
    x = X[i]
    pred = getPred(x, w)
    w = w + ((y[i]-pred)*x) # update w
    _iter += 1
    error, misclassifieds = getError(X, w, y)
    
    # store best w in pocket
    if error < error_best:
        error_best = error
        w_best = w

    plt.title(f"Iteration {_iter}: Misclassified {error}/{n}")
    plotPLA(plotX, w, y, pause=0.5)

w = w_best
if(error_best == 0):
    plt.title(f"Correctly classified all points after {_iter} iterations")
else:
    plt.title(f"Failed to classify all points after {_iter} iterations\nBest weights correctly classified {n-error_best}/{n} points")

plotPLA(plotX, w, y)

