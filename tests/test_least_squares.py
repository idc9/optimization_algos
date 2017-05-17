import unittest
from least_squares import *
from synthetic_data import *


class LeastSquaresTests(unittest.TestCase):
    """test least squares"""
    def setUp(self):

        # sample least squares data
        n = 100
        d = 10
        X, y = generate_LS_data(n, d, seed=34234)
        self.X = X
        self.y = y

        self.model = LeastSquares(X, y)

        self.test_point = np.random.normal(loc=0, scale=10, size=d)

        self.epsilon = 1e-10

    def test_individual_lik(self):
        """
        For ERM problems make sure the mean of the individual likelihoods
        is equal to the full function value
        """
        # mean likelihood
        like_mean = np.mean([self.model.f(self.test_point, i) for i in range(self.model.n)])

        # 2-norm of difference
        norm_diff = np.linalg.norm(like_mean - self.model.F(self.test_point))

        self.assertTrue(norm_diff < self.epsilon)

    def test_individual_lik_grad(self):
        """
        For ERM problems make sure the mean of the individual likelihood
        gradients is equal to the full function value
        """
        # mean likelihood
        like_grad_mean = (1.0/self.model.n)*sum([self.model.grad_f(self.test_point, i) for i in range(self.model.n)])

        # 2-norm of difference
        norm_diff = np.linalg.norm(like_grad_mean - self.model.grad_F(self.test_point))

        self.assertTrue(norm_diff < self.epsilon)



unittest.main()
