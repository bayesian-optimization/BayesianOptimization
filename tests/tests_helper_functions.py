from __future__ import print_function
from __future__ import division
import unittest
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import sys
sys.path.append("./")
from bayes_opt.helpers import UtilityFunction, acq_max


def get_globals():
    X = np.array([
        [0.00, 0.00],
        [0.99, 0.99],
        [0.00, 0.99],
        [0.99, 0.00],
        [0.50, 0.50],
        [0.25, 0.50],
        [0.50, 0.25],
        [0.75, 0.50],
        [0.50, 0.75],
    ])

    def get_y(X):
        return -(X[:, 0] - 0.3) ** 2 - 0.5 * (X[:, 1] - 0.6)**2 + 2
    y = get_y(X)

    mesh = np.dstack(
        np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
    ).reshape(-1, 2)

    GP = GaussianProcessRegressor(
        kernel=Matern(),
        n_restarts_optimizer=25,
    )
    GP.fit(X, y)

    return {'x': X, 'y': y, 'gp': GP, 'mesh': mesh}


def brute_force_maximum(MESH, GP, kind='ucb', kappa=1.0, xi=1e-6):
    uf = UtilityFunction(kind=kind, kappa=kappa, xi=xi)

    mesh_vals = uf.utility(MESH, GP, 2)
    max_val = mesh_vals.max()
    max_arg_val = MESH[np.argmax(mesh_vals)]

    return max_val, max_arg_val


GLOB = get_globals()
X, Y, GP, MESH = GLOB['x'], GLOB['y'], GLOB['gp'], GLOB['mesh']


class TestMaximizationOfAcquisitionFunction(unittest.TestCase):

    def setUp(self, kind='ucb', kappa=1.0, xi=1e-6):
        self.util = UtilityFunction(kind=kind, kappa=kappa, xi=xi)
        self.episilon = 1e-2
        self.y_max = 2.0

    def test_acq_max_function_with_ucb_algo(self):
        self.setUp(kind='ucb', kappa=1.0, xi=1.0)
        max_arg = acq_max(
            self.util.utility, GP, self.y_max, bounds=np.array([[0, 1], [0, 1]])
        )
        _, brute_max_arg = brute_force_maximum(MESH, GP)

        self.assertTrue( all(abs(brute_max_arg - max_arg) < self.episilon))

    def test_ei_max_function_with_ucb_algo(self):
        self.setUp(kind='ei', kappa=1.0, xi=1e-6)
        max_arg = acq_max(
            self.util.utility, GP, self.y_max, bounds=np.array([[0, 1], [0, 1]])
        )
        _, brute_max_arg = brute_force_maximum(MESH, GP, kind='ei')

        self.assertTrue( all(abs(brute_max_arg - max_arg) < self.episilon))


if __name__ == '__main__':
    unittest.main()
