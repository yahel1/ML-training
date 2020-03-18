import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
import os
from scipy.sparse import csr_matrix
from sklearn import svm


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


if __name__ == '__main__':
    train_features, train_labels = load_svmlight_file('twofeature.txt')
    x0, x1 = csr_matrix.toarray(train_features).T
    colors = ['red' if label == 1 else 'green' for label in train_labels]
    xx, yy = make_meshgrid(x0, x1)
    fig, ax = plt.subplots(1, 2)
    indC = 0
    for C in [1, 100]:
        clf = svm.SVC(kernel='linear', C=C)
        clf.fit(np.array([x0, x1]).T, train_labels)
        plot_contours(ax[indC], clf, xx, yy)
        ax[indC].scatter(x0, x1, c=train_labels, cmap=plt.cm.coolwarm, edgecolors='k')
        ax[indC].set_title(f'C={C}')
        indC += 1
