import numpy as np
from scipy.special import expit


class LogisticRegression(object):
    """
    Logistic regression

    min_x sum_i log(1 + exp(b_i *a_i^Tx))

    A in R^(n x d) assumes the last column is all ones for the intercet
    b =+/-1 in R^n
    x in R^d
    """

    def __init__(self, X, y):

        # data
        self.X = y
        self.y = y

        self.d = X.shape[1]
        self.n = X.shape[0]

        # lipshitz constant
        self.L_F = .25 * np.linalg.norm(np.dot(X.T, X))

    def F(self, beta):
        return -1*logistic_loss(self.X, self.y, beta)

    def grad_F(self, beta):
        return -1*logistic_loss_grad(self.X, self.y, beta)

    def f(self, beta, i):
        return -1*self.n*logistic_loss(self.X[i, :], self.y[i], beta)

    def grad_f(self, beta, i):
        return -1*self.n*logistic_loss_grad(self.X[i, :], self.y[i], beta)


def logistic_loss(X, y, beta):
    """ sum_i log(1 + exp(y_i *x_i^T beta))"""
    return np.sum(np.log(1 + np.exp(np.multiply(y, np.dot(X, beta)))))


def logistic_loss_grad(X, y, beta):
    # E = expit(-1*np.multiply(y, np.dot(X, beta)))
    # Eb = np.multiply(E, y).T
    # return np.sum(np.dot(np.diag(Eb), X), axis=0)

    # if y is a scalar need a different format
    if not hasattr(y, "__len__"):
        y01 = int(0 < y)
        p = expit(np.dot(X, beta))
        grad = X * (y01 - p)
    else:
        p = expit(np.dot(X, beta))
        y01 = [1 if label > 0 else 0 for label in y]
        grad = np.dot(X.T, y01 - p)

    return grad
