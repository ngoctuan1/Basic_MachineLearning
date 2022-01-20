import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def sigmoid(X, w):
    z = X.dot(w)
    return 1/(1+np.exp(-z))


def loss(X, y, w, lam):
    a = sigmoid(X, w)
    return -np.mean(y*np.log(a) - (1-y)*np.log(1-a)) + 0.5*lam/X.shape[0]*np.sum(w**2)


def gradient(x, y, w, lam):
    a = sigmoid(x, w)
    return ((a-y)*x) + lam*w

def logistic_regression(X, y, w_init, lam = 0.0001, lr=0.1, max_epochs = 1000):
    w = w_old = w_init
    N, d = X.shape[:2]
    loss_hist = [loss(X, y, w_init, lam)]
    ep = 0
    while ep < max_epochs:
        ep+=1
        mix_ids = np.random.permutation(N)
        for id in mix_ids:
            xi = X[id]
            yi = y[id]
            w = w - lr*gradient(xi, yi, w, lam)
        
        loss_hist.append(loss(X, y, w, lam))
        if np.linalg.norm(w - w_old) / d < 1e-10:
            break
        w_old = w
    return w, loss_hist


# X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
#               2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
# y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# Xbar = np.concatenate((X, np.ones((X.shape[0], 1), dtype=np.uint8)), axis=1)
# w_init = np.random.randn(Xbar.shape[1])
# lam = 0.0001
# w, loss_hist = logistic_regression(Xbar, y, w_init, lam)

# model  = LogisticRegression(tol = 1e-10, max_iter=1000)
# model.fit(X, y)

# plt.subplot(2,1,1)
# plt.plot(X[y==0], y[y==0], 'ro', label = "y = 0")
# plt.plot(X[y==1], y[y==1], 'bs', label = "y = 1")
# plt.legend()

# plt.subplot(2,1,2)
# plt.plot(X[y == 0], y[y == 0], 'ro', label="y = 0")
# plt.plot(X[y == 1], y[y == 1], 'bs', label="y = 1")
# plt.plot(X, sigmoid(Xbar, w), 'g', label = "Use GD")
# plt.plot(X, model.predict_proba(X)[:,1], 'y', label = "Use sklearn")
# plt.legend()

# plt.show()
