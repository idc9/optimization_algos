import numpy as np


def accelerated_gradient_descent(model, eta, max_iterations=1e4, epsilon=1e-5,
                                 beta_start=None):
    """
    Nesterov's accelerated gradient descent

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

    y_current = beta_current
    t_current = 1.0

    # history
    beta_history = []

    for k in range(int(max_iterations)):
        # history
        beta_history.append(beta_current)

        # gradient update
        t_next = .5*(1 + np.sqrt(1 + 4*t_current**2))
        beta_next = y_current - eta * grad_F(y_current)
        y_next = beta_next + (t_current - 1.0)/(t_next)*(beta_next - beta_current)

        # relative error stoping condition
        if np.linalg.norm(beta_next - beta_current) <= epsilon*np.linalg.norm(beta_current):
            break
            # if np.linalg.norm(beta_next) <= epsilon:

        # restarting strategies
        if np.dot(y_current - beta_next, beta_next - beta_current) > 0:
            y_next = beta_next
            t_next = 1
            # if k %% k_restart == 0:

        beta_current = beta_next
        y_current = y_next
        t_current = t_next

    print 'accelerated GD finished after ' + str(k) + ' iterations'

    return {'solution': beta_current,
            'beta_history': beta_history}
