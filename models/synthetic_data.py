import numpy as np
from scipy.special import expit


def generate_LS_data(n, d, seed=None):
    """ samples random LS data"""

    if seed:
        np.random.seed(3312)

    X = np.random.normal(loc=0, scale=1, size=(n, d))
    y = np.random.normal(loc=0, scale=1, size=n)

    return X, y


def generate_logreg_data(n, d, seed=None):
    """samples random logistic regression data"""
    if seed:
        np.random.seed(3312)

    # X data with intercept
    X = np.random.normal(loc=0, scale=1, size=(n, d-1))
    X = np.array(np.bmat([X, np.ones((n, 1))]))  # add i ntercept to design matrix

    # true beta
    beta_platon = np.random.normal(loc=0, scale=1, size=d)

    # sample data
    prob = expit(np.dot(A, beta_platon))
    unif = np.random.sample(n)
    y = np.array([1 if unif[i] < prob[i] else -1 for i in range(n)])  # class labels

    return X, y, beta_platon
