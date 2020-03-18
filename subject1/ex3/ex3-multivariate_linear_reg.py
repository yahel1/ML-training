import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.genfromtxt('ex3x.dat')
y = np.genfromtxt('ex3y.dat')

m = len(y)
x = np.concatenate((np.ones([m, 1]), x), axis=1)
theta_normalEQ = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
size, N_bedrooms = 1650, 3
print(f'price: {np.dot(theta_normalEQ, np.array([1, size, N_bedrooms]))/1000:.3f} k$')

sigma = np.std(x, axis=0)
mu = np.mean(x, axis=0)
for ind in [1, 2]:
    x[:, ind] = (x[:, ind] - mu[ind]) / sigma[ind]

n_iter = 100
plt.figure()
for alpha in [0.01, 0.03, 0.1, 0.3, 1, 1.3]:
    theta = np.zeros([len(mu), 1])
    J = np.zeros(n_iter)
    for i in range(n_iter):
        h = np.dot(theta.T, x.T)
        J[i] = 1 / 2 / m * np.sum(np.square(h - y))
        for j in range(len(theta)):
            theta[j] -= alpha/m*np.sum((h - y)*x[:, j])
    if alpha == 1:
        print(theta)
        print(f'price: {np.dot(theta.T, np.array([1, (size-mu[1])/sigma[1], (N_bedrooms-mu[2])/sigma[2]]))[0] / 1000:.3f} k$')
    plt.plot(np.arange(n_iter), J, '-', label=f'$\\alpha$={alpha:.3f}')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.xlim([0, n_iter])
plt.legend()
