import numpy as np

from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt import BayesianOptimization


def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1


def test_bound_x_maximize():
    bounds_transformer = SequentialDomainReductionTransformer()
    pbounds = {'x': (-10, 10), 'y': (-10, 10)}
    n_iter = 10

    standard_optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )

    standard_optimizer.maximize(
        init_points=2,
        n_iter=n_iter,
    )

    mutated_optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
        bounds_transformer=bounds_transformer
    )

    mutated_optimizer.maximize(
        init_points=2,
        n_iter=n_iter,
    )

    assert len(standard_optimizer.space) == len(mutated_optimizer.space)
    assert not (standard_optimizer._space.float_bounds ==
                mutated_optimizer._space.float_bounds).any()

def test_minimum_window_is_kept():
    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=1.0)
    pbounds = {'x': (-0.5, 0.5), 'y': (-1.0, 0.0)}
    mutated_optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=0,
        random_state=1,
        bounds_transformer=bounds_transformer
    )

    mutated_optimizer.maximize(
        init_points=2,
        n_iter=10,
    )
    window_width = np.diff(bounds_transformer.bounds)
    assert np.isclose(np.min(window_width), 1.0)


def test_minimum_window_array_is_kept():
    window_ranges = [1.0, 0.5]
    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=window_ranges)
    pbounds = {'x': (-0.5, 0.5), 'y': (-1.0, 0.0)}
    mutated_optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=0,
        random_state=1,
        bounds_transformer=bounds_transformer
    )

    mutated_optimizer.maximize(
        init_points=2,
        n_iter=10,
    )
    window_widths = np.diff(bounds_transformer.bounds)
    assert np.all(np.isclose(np.squeeze(np.min(window_widths, axis=0)), window_ranges))
