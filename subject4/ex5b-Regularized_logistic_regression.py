import matplotlib.pyplot as plt
import numpy as np


def plot_contours(ax, x, theta, degree):
    Npts = 200
    z = np.zeros([Npts, Npts])
    x0, x1 = x[:, 0], x[:, 1]
    u1 = np.linspace(min(x0), max(x0), Npts)
    v1 = np.linspace(min(x1), max(x1), Npts)
    for iu in range(Npts):
        for jv in range(Npts):
            uv_features = np.array([u1[iu] ** (i - j) * v1[jv] ** j for i in range(degree + 1) for j in range(i + 1)])
            z[jv, iu] = np.dot(uv_features, theta)
    out = ax.contour(u1, v1, z, levels=0)
    return out


if __name__ == '__main__':
    x = np.genfromtxt('ex5Logx.dat', dtype=None, delimiter=',')
    y = np.genfromtxt('ex5Logy.dat', dtype=None, delimiter=' ')
    m = len(y)
    u, v = x[:, 0], x[:, 1]
    pos, neg = np.where(y == 1), np.where(y == 0)

    degree = 6
    features = np.array([u**(i-j) * v**j for i in range(degree+1) for j in range(i+1)])
    nf = features.shape[0]
    Id = np.eye(nf)
    Id[0] = 0
    n_iter = 20
    indR = 0
    figR, axR = plt.subplots(3, 1)
    figJ, axJ = plt.subplots(3, 1)
    for regularization_param in [0, 1, 10]:
        J = np.zeros(n_iter)
        theta = np.zeros(nf)
        for i in range(n_iter):
            h = 1/(1 + np.exp(-np.dot(theta.T, features)))
            J[i] = 1 / m * np.sum(-y*np.log(h) - (1-y)*np.log(1-h)) + regularization_param/2/m*sum(theta[1:]**2)
            gradJ = 1 / m * sum((h - y).reshape([m, 1])*features.T) + theta*regularization_param/m
            H = 1 / m * np.dot(np.dot(features, np.diag(h) * np.diag(1-h)), features.T) + regularization_param/m*Id
            theta -= np.dot(np.linalg.inv(H), gradJ)
            axJ[indR].plot(i, J[i], 'ob')

        axR[indR].plot(u[pos], v[pos], '+', label='y=1')
        axR[indR].plot(u[neg], v[neg], 'o', label='y=0')
        axR[indR].legend(fontsize=12)
        plot_contours(axR[indR], x, theta, degree)
        axR[indR].set_title(f'$\lambda$={regularization_param}')
        indR += 1
