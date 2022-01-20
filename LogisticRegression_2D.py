import numpy as np
import matplotlib.pyplot as plt
from  scipy.spatial.distance import cdist
from LogisticRegression import logistic_regression,sigmoid

# np.random.seed(42)
means = [[2, 2], [4, 2]]
cov = [[.7, 0], [0, .7]]
N = 20
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.concatenate((X0,X1), axis=0)
y = np.concatenate((np.zeros((1, N)), np.ones((1,N))), axis=1).T

# Xbar
Xbar = np.concatenate((np.ones((X.shape[0], 1)), X),axis =1)
eta = .05
d = Xbar.shape[1]
w_init = np.random.randn(d)
lam = 0.0001
w,loss_hist = logistic_regression(Xbar, y, w_init, lam, eta,max_epochs=1000)


xm = np.linspace(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1, 200)
ym = np.linspace(np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1, 200)
xm, ym = np.meshgrid(xm, ym)
zm = 1/(1+np.exp(-(w[0] + w[1]*xm + w[2]*ym)))


CS = plt.contourf(xm, ym, zm, 200, cmap='jet')
plt.plot(X0[:, 0], X0[:, 1], 'bs', markersize=8, alpha=1, markeredgecolor="w")
plt.plot(X1[:, 0], X1[:, 1], 'ro', markersize=8, alpha=1, markeredgecolor="w")
plot_x = np.array([-1, 6], dtype=np.float32)
plot_y = (-1/w[2])*(w[1]*plot_x + w[0])
plt.plot(plot_x, plot_y, 'r')
plt.ylim(0, 4)
plt.xlim(0, 5)

# hide ticks
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

plt.xlabel('x_1', fontsize = 20)
plt.ylabel('y_1', fontsize = 20)
plt.show()

