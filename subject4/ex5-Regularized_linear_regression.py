import matplotlib.pyplot as plt
import numpy as np

x = np.genfromtxt('ex5Linx.dat', dtype=None, delimiter=' ')
y = np.genfromtxt('ex5Liny.dat', dtype=None, delimiter=' ')
plt.plot(x, y, 'o', label='Training data')
m = len(x)
n = 6
features = np.array([x**i for i in range(n)]).T
Id = np.eye(n)
Id[0] = 0
indR = 0
theta = [None]*3
for regularization_param in [0, 1, 10]:
    theta[indR] = np.dot(np.linalg.inv((np.dot(features.T, features) + regularization_param*Id)), np.dot(features.T, y))
    x_plt_fit = np.linspace(min(x), max(x), 100)
    plt.plot(x_plt_fit, sum([x_plt_fit**i*theta[indR][i] for i in range(n)]),
             '--', label=f'$\lambda$={regularization_param}')
    indR += 1
norm_theta = np.linalg.norm(theta, axis=1)
plt.legend(fontsize=16)
plt.show()
plt.title('Regularized linear regression', fontsize=16)

