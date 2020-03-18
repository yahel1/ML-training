from sklearn.datasets import load_svmlight_file
from sklearn import svm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import numpy as np


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
    train_features, train_labels = load_svmlight_file('ex8a.txt')
    clf = svm.SVC(kernel='rbf', gamma=100)
    clf.fit(train_features, train_labels)

    x0, x1 = csr_matrix.toarray(train_features).T
    xx, yy = make_meshgrid(x0, x1)
    fig, ax = plt.subplots(1, 1)
    plot_contours(ax, clf, xx, yy)
    ax.scatter(x0, x1, c=train_labels, cmap=plt.cm.coolwarm, edgecolors='k')
    ax.set_xlim(x0.min(), x0.max())
    ax.set_ylim(x1.min(), x1.max())
    ax.set_title(f'$\gamma$=100', fontsize=18)