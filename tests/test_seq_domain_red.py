import numpy as np
import pytest

from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt import BayesianOptimization
from bayes_opt.target_space import TargetSpace

def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1


def test_bound_x_maximize():

    class Tracker:
        def __init__(self):
            self.start_count = 0
            self.step_count = 0
            self.end_count = 0

        def update_start(self, event, instance):
            self.start_count += 1

        def update_step(self, event, instance):
            self.step_count += 1

        def update_end(self, event, instance):
            self.end_count += 1

        def reset(self):
            self.__init__()

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
    assert not (standard_optimizer._space.bounds ==
                mutated_optimizer._space.bounds).any()

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

def test_trimming_bounds():
    """Test if the bounds are trimmed correctly within the bounds"""
    def dummy_function(x1, x2, x3, x4, x5):
        return 0.0

    min_window = 1.0
    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=min_window)
    pbounds  = {
        'x1': (-1, 0.6),
        'x2': (-1, 0.5),
        'x3': (-0.4, 0.6),
        'x4': (0.3, 1.3),
        'x5': (-1, 0.8),
    }
    target_sp = TargetSpace(target_func=dummy_function, pbounds=pbounds)
    bounds_transformer.initialize(target_sp)
    new_bounds = np.concatenate((np.ones((5, 1)) * 0.1, np.ones((5, 1))), axis=1)
    global_bounds = np.asarray(list(pbounds.values()))
   
    trimmed_bounds = bounds_transformer._trim(new_bounds, global_bounds)
    # check that the bounds are trimmed to the minimum window
    # raises ValueError if the bounds are not trimmed correctly
    bounds_transformer._window_bounds_compatiblity(trimmed_bounds)


def test_exceeded_bounds():
    """Raises Value Error if the bounds are exceeded."""
    window_ranges = [1.01, 0.72]
    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=window_ranges)
    pbounds = {'x': (-0.5, 0.5), 'y': (-0.7, 0.0)}
    with pytest.raises(ValueError):
        _ = BayesianOptimization(
                f=black_box_function,
                pbounds=pbounds,
                verbose=0,
                random_state=1,
                bounds_transformer=bounds_transformer
            )