from bayes_opt import BayesianOptimization
from bayes_opt.helpers import ensure_rng
import numpy as np


def test_bayes_opt_demo():
    """
    pytest tests/test_bayesian_optimization.py::test_bayes_opt_demo

    See Also
    --------
    https://github.com/fmfn/BayesianOptimization/blob/master/examples/exploitation%20vs%20exploration.ipynb
    """
    random_state = ensure_rng(0)
    xs = np.linspace(-2, 10, 1000)
    f = np.exp(-(xs - 2)**2) + np.exp(-(xs - 6)**2 / 10) + 1 / (xs**2 + 1)
    bo = BayesianOptimization(f=lambda x: f[int(x)],
                              pbounds={'x': (0, len(f) - 1)},
                              random_state=random_state,
                              verbose=0)
    gp_params = {'alpha': 1e-5, 'n_restarts_optimizer': 2}
    # Change aquisition params to speedup optimization for testing purposes
    bo._acqkw['n_iter'] = 5
    bo._acqkw['n_warmup'] = 1000
    bo.maximize(init_points=10, n_iter=5, acq='ucb', kappa=5, **gp_params)
    res = bo.space.max_point()
    max_params = res['max_params']
    max_val = res['max_val']

    ratio = max_val / f.max()
    assert max_val > 1.1, 'got {}, but should be > 1'.format(max_val)
    assert ratio > .9, 'got {}, should be better than 90% of max val'.format(ratio)

    assert max_params['x'] > 300, 'should be in a peak area (around 300)'
    assert max_params['x'] < 400, 'should be in a peak area (around 300)'


def test_only_random():
    random_state = ensure_rng(0)
    xs = np.linspace(-2, 10, 1000)
    f = np.exp(-(xs - 2)**2) + np.exp(-(xs - 6)**2 / 10) + 1 / (xs**2 + 1)
    bo = BayesianOptimization(f=lambda x: f[int(x)],
                              pbounds={'x': (0, len(f) - 1)},
                              random_state=random_state,
                              verbose=0)
    bo.init(20)
    res = bo.space.max_point()
    max_params = res['max_params']
    max_val = res['max_val']

    assert max_val > 1.0, 'function range is ~.2 - ~1.4, should be above 1.'
    assert max_val / f.max() > .8, 'should be better than 80% of max val'

    assert max_params['x'] > 200, 'should be in a peak area (around 300)'
    assert max_params['x'] < 500, 'should be in a peak area (around 300)'


def test_explore_lazy():
    random_state = ensure_rng(0)
    xs = np.linspace(-2, 10, 1000)
    f = np.exp(-(xs - 2)**2) + np.exp(-(xs - 6)**2 / 10) + 1 / (xs**2 + 1)
    bo = BayesianOptimization(f=lambda x: f[int(x)],
                              pbounds={'x': (0, len(f) - 1)},
                              random_state=random_state,
                              verbose=0)
    bo.explore({'x': [f.argmin()]}, eager=False)
    assert len(bo.space) == 0
    assert len(bo.init_points) == 1

    # Note we currently expect lazy explore to override points
    # This may not be the case in the future.
    bo.explore({'x': [f.argmax()]}, eager=False)
    assert len(bo.space) == 0
    assert len(bo.init_points) == 1

    bo.maximize(init_points=0, n_iter=0, acq='ucb', kappa=5)

    res = bo.space.max_point()
    max_params = res['max_params']
    max_val = res['max_val']

    assert max_params['x'] == f.argmax()
    assert max_val == f.max()


def test_explore_eager():
    random_state = ensure_rng(0)
    xs = np.linspace(-2, 10, 1000)
    f = np.exp(-(xs - 2)**2) + np.exp(-(xs - 6)**2 / 10) + 1 / (xs**2 + 1)
    bo = BayesianOptimization(f=lambda x: f[int(x)],
                              pbounds={'x': (0, len(f) - 1)},
                              random_state=random_state,
                              verbose=0)
    bo.explore({'x': [f.argmin()]}, eager=True)
    assert len(bo.space) == 1
    assert len(bo.init_points) == 0

    # Note we currently expect lazy explore to override points
    # This may not be the case in the future.
    bo.explore({'x': [f.argmax()]}, eager=True)
    assert len(bo.space) == 2
    assert len(bo.init_points) == 0

    res = bo.space.max_point()
    max_params = res['max_params']
    max_val = res['max_val']

    assert max_params['x'] == f.argmax()
    assert max_val == f.max()


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_bayesian_optimization.py
    """
    import pytest
    pytest.main([__file__])
