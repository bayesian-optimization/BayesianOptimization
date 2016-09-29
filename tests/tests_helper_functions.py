from __future__ import print_function
from __future__ import division
import unittest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class TestMaximizationOfAcquisitionFunction(unittest.TestCase):

    def setUp(self, msg=""):
        self.gp = GaussianProcessRegressor(
            kernel=Matern(),
            n_restarts_optimizer=25,
        )

        X = np.array([
            [1, 2],
            [2, 3],
            [0, -1],
        ])

        y = X[:, 0]**2 - 2 * X[:, 1]

    def test_something(self):
        pass


if __name__ == '__main__':
    unittest.main()
