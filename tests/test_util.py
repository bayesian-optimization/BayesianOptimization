import numpy as np

from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction, acq_max, load_logs, ensure_rng

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor


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
        np.meshgrid(np.arange(0, 1, 0.005), np.arange(0, 1, 0.005))
    ).reshape(-1, 2)

    GP = GaussianProcessRegressor(
        kernel=Matern(),
        n_restarts_optimizer=25,
    )
    GP.fit(X, y)

    return {'x': X, 'y': y, 'gp': GP, 'mesh': mesh}


def brute_force_maximum(MESH, GP, kind='ucb', kappa=1.0, xi=1.0):
    uf = UtilityFunction(kind=kind, kappa=kappa, xi=xi)

    mesh_vals = uf.utility(MESH, GP, 2)
    max_val = mesh_vals.max()
    max_arg_val = MESH[np.argmax(mesh_vals)]

    return max_val, max_arg_val


GLOB = get_globals()
X, Y, GP, MESH = GLOB['x'], GLOB['y'], GLOB['gp'], GLOB['mesh']


def test_acq_with_ucb():
    util = UtilityFunction(kind="ucb", kappa=1.0, xi=1.0)
    episilon = 1e-2
    y_max = 2.0

    max_arg = acq_max(
        util.utility,
        GP,
        y_max,
        bounds=np.array([[0, 1], [0, 1]]),
        random_state=ensure_rng(0),
        n_iter=20
    )
    _, brute_max_arg = brute_force_maximum(MESH, GP, kind='ucb', kappa=1.0, xi=1.0)

    assert all(abs(brute_max_arg - max_arg) < episilon)


def test_acq_with_ei():
    util = UtilityFunction(kind="ei", kappa=1.0, xi=1e-6)
    episilon = 1e-2
    y_max = 2.0

    max_arg = acq_max(
        util.utility,
        GP,
        y_max,
        bounds=np.array([[0, 1], [0, 1]]),
        random_state=ensure_rng(0),
        n_iter=200,
    )
    _, brute_max_arg = brute_force_maximum(MESH, GP, kind='ei', kappa=1.0, xi=1e-6)

    assert all(abs(brute_max_arg - max_arg) < episilon)


def test_acq_with_poi():
    util = UtilityFunction(kind="poi", kappa=1.0, xi=1e-4)
    episilon = 1e-2
    y_max = 2.0

    max_arg = acq_max(
        util.utility,
        GP,
        y_max,
        bounds=np.array([[0, 1], [0, 1]]),
        random_state=ensure_rng(0),
        n_iter=200,
    )
    _, brute_max_arg = brute_force_maximum(MESH, GP, kind='poi', kappa=1.0, xi=1e-4)

    assert all(abs(brute_max_arg - max_arg) < episilon)


def test_logs():
    import pytest
    def f(x, y):
        return -x ** 2 - (y - 1) ** 2 + 1

    optimizer = BayesianOptimization(
        f=f,
        pbounds={"x": (-2, 2), "y": (-2, 2)}
    )
    assert len(optimizer.space) == 0

    load_logs(optimizer, "./tests/test_logs.json")
    assert len(optimizer.space) == 5

    load_logs(optimizer, ["./tests/test_logs.json"])
    assert len(optimizer.space) == 5

    other_optimizer = BayesianOptimization(
        f=lambda x: -x ** 2,
        pbounds={"x": (-2, 2)}
    )
    with pytest.raises(ValueError):
        load_logs(other_optimizer, ["./tests/test_logs.json"])


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_target_space.py
    """
    import pytest
    pytest.main([__file__])
