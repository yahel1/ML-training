import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from sklearn import svm


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
