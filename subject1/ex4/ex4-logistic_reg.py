import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.genfromtxt('ex4x.dat')
y = np.genfromtxt('ex4y.dat')

m = len(y)
x = np.concatenate((np.ones([m, 1]), x), axis=1)
pos = np.where(y == 1)
neg = np.where(y == 0)

plt.figure()
plt.plot(np.squeeze(x[pos, 1]), np.squeeze(x[pos, 2]), '+')
plt.plot(np.squeeze(x[neg, 1]), np.squeeze(x[neg, 2]), 'o')

theta = np.zeros(x.shape[1])
n_iter = 6
J = np.zeros(n_iter)
for i in range(n_iter):
    h = 1/(1 + np.exp(-np.dot(theta.T, x.T)))
    J[i] = 1/m * np.sum(-y*np.log(h) - (1-y)*np.log(1-h))
    gradJ = 1/m * sum((h - y).reshape([80, 1])*x)
    H = 1/m * sum([h[j]*(1-h[j])*np.expand_dims(x[j, :], axis=1)*np.expand_dims(x[j, :], axis=1).T for j in range(m)])
    theta -= np.dot(np.linalg.inv(H), gradJ)

plt.plot(x[:, 1], -(theta[0] + theta[1]*x[:, 1])/theta[2], '--')
plt.xlabel('exam 1 score')
plt.ylabel('exam 2 score')
plt.figure()
plt.plot(np.arange(n_iter), J, 'o-')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
score1, score2 = 20, 80
cost = - (theta[0] + theta[1]*score1 + theta[2]*score2)
print(f'probability to 0 if score1=20 & score2=80: {cost:.3}')

