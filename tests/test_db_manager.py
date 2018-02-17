from bayes_opt import BayesianOptimization
from bayes_opt.helpers import ensure_rng, PrintLog
import numpy as np
import pytest


def target(x, y, z):
    return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1 / (x**2 + 1) + \
        np.exp(-(y - 2)**2) + np.exp(-(y - 6)**2/10) + 1 / (y**2 + 1) + \
        np.exp(-(z - 2)**2) + np.exp(-(z - 6)**2/10) + 1 / (z**2 + 1)


def test_db_manager_dimension():
    """
    pytest tests/test_db_manager.py::test_db_manager

    See Also
    --------
    https://github.com/fmfn/BayesianOptimization/blob/master/examples/database_interactions.ipynb
    """

    random_state = ensure_rng(0)
    gp_params = {'alpha': 1e-5, 'n_restarts_optimizer': 2}
    bo = BayesianOptimization(target,
                              pbounds={'x': (-2, 10), 'y': (-2, 10), 'z': (-2, 10)},
                              random_state=random_state,
                              verbose=1)

    # Change aquisition params to speedup optimization for testing purposes
    bo._acqkw['n_iter'] = 5
    bo._acqkw['n_warmup'] = 1000

    bo.init_db('sqlite:///target_space.db')
    bo.maximize(init_points=2, n_iter=2, acq='ucb', kappa=5, **gp_params)
    bo.save()
    for i in range(1):
        bo = BayesianOptimization(target,
                                  pbounds={'x': (-2, 10), 'y': (-2, 10), 'z': (-2, 10)},
                                  random_state=random_state,
                                  verbose=1)
        bo._acqkw['n_iter'] = 5
        bo._acqkw['n_warmup'] = 1000
        bo.init_db('sqlite:///target_space.db')
        bo.load()
        bo.maximize(init_points=0, n_iter=2, acq='ucb', kappa=5, **gp_params)
        bo.save()
        bo.print_summary()


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_bayesian_optimization.py
    """
    pytest.main([__file__])
