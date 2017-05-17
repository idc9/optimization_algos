import numpy as np


def gradient_descent(model, eta, max_iterations=1e4, epsilon=1e-5,
                     beta_start=None):
    """
    Gradient descent

    Parameters
    ----------
    model: optimization model object
    eta: learning rate
    max_iterations: maximum number of gradient iterations
    epsilon: tolerance for stopping condition
    beta_start: where to start (otherwise random)

    Output
    ------
    solution: final beta value
    beta_history: beta values from each iteration
    """

    # data from model
    grad_F = model.grad_F
    d = model.d
    # F = model.F

    # initialization
    if beta_start:
        beta_current = beta_start
    else:
        beta_current = np.random.normal(loc=0, scale=1, size=d)

    # keep track of history
    beta_history = []

    for k in range(int(max_iterations)):

        beta_history.append(beta_current)

        # gradient update
        beta_next = beta_current - eta * grad_F(beta_current)

        # relative error stoping condition
        if np.linalg.norm(beta_next - beta_current) <= epsilon*np.linalg.norm(beta_current):
            #  if np.linalg.norm(beta_next) <= epsilon:
            break

        beta_current = beta_next

    print 'GD finished after ' + str(k) + ' iterations'

    return {'solution': beta_current,
            'beta_history': beta_history}
