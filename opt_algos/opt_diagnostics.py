import numpy as np
import matplotlib.pyplot as plt


def opt_history(model, beta_history, beta_solution):
    """
    Computes optimization history

    Parameters
    ----------
    model: optimization model
    beta_history: history of beta values

    Returns
    -------
    fun_history: function values
    grad_history: 2-norm of gradient
    beta_error: 2-norm error of beta from solution where solution is either
    provided or computed based on the min function value

    fun_history, grad_history, beta_error = opt_history(model, beta_history)
    """

    fun_history = [model.F(b) for b in beta_history]
    grad_history = [np.linalg.norm(model.grad_F(b)) for b in beta_history]

    # beta_solution = model.get_solution()
    # if not beta_solution:
    # beta giving min function value
    # beta_solution = beta_history[np.argmin(fun_history)]

    beta_error = [np.linalg.norm(beta_solution - b) for b in beta_history]

    return fun_history, grad_history, beta_error


def plot_opt_path(beta_history, model, beta_solution, opt_algo=''):

    # model history
    fun_history, grad_history, beta_error = opt_history(model, beta_history, beta_solution)

    plt.figure(figsize=[15, 5])

    # absolutes
    plt.subplot(2, 3, 1)
    plt.plot(fun_history)
    plt.ylabel('function value')
    plt.xlabel('iteration')
    plt.title(opt_algo + ' for ' + model.name)

    plt.subplot(2, 3, 2)
    plt.plot(beta_error)
    plt.ylabel('||beta - beta*||')
    plt.xlabel('iteration')

    plt.subplot(2, 3, 3)
    plt.plot(grad_history)
    plt.ylabel('grad F')
    plt.xlabel('iteration')

    # # differences
    # plt.subplot(2,3,4)
    # plt.plot(np.diff(fun_history))
    # plt.ylabel('function value diff')
    # plt.xlabel('iteration')

    # plt.subplot(2,3,5)
    # plt.plot(np.diff(beta_error))
    # plt.ylabel('||beta - beta*|| diff')
    # plt.xlabel('iteration')

    # plt.subplot(2,3,6)
    # plt.plot(np.diff(grad_history))
    # plt.ylabel('grad F diff')
    # plt.xlabel('iteration')
