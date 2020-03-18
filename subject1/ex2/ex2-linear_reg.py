import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.genfromtxt('ex2x.dat', dtype=None, delimiter=' ')
y = np.genfromtxt('ex2y.dat', dtype=None, delimiter=' ')
plt.figure()
plt.plot(x, y, 'o', label='Training data')
plt.ylabel('Height in meters')
plt.xlabel('Age in years')

m = len(y)
x = np.concatenate((np.ones([m, 1]), x.reshape([m, 1])), axis=1)
alpha = 0.07
theta = np.zeros([2, 1])
n_iter = 10000
for i in range(n_iter):
    h = np.dot(theta.T, x.T)
    for j in range(len(theta)):
        theta[j] -= alpha/m*np.sum((h - y)*x[:, j])

plt.plot(x[:, 1], theta[0] + x[:, 1]*theta[1], '-', label='Linear regression')
ages = [3.5, 7]
for age in ages:
    print(f'age {age}, predicted height is: {theta[1]*age + theta[0]}')

N = 100
J_vals = np.zeros([N, N])
theta0_vals = np.linspace(-3, 3, N)
theta1_vals = np.linspace(-1, 1, N)
for i in range(N):
    for j in range(N):
         J_vals[i, j] = 1/2/m*np.sum(np.square(theta0_vals[i] + theta1_vals[j]*x[:, 1] - y))

minJ_inds = np.unravel_index(np.argmin(J_vals, axis=None), J_vals.shape)
plt.plot(x[:, 1], theta0_vals[minJ_inds[0]] + x[:, 1]*theta1_vals[minJ_inds[1]], '-', label=r'min J(${\Theta}$)')
plt.legend()
fig3D = plt.figure()
ax = fig3D.add_subplot(111, projection='3d')
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals)

