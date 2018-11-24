import pytest
import numpy as np
from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization


def target_func(**kwargs):
    # arbitrary target func
    return sum(kwargs.values())


PBOUNDS = {'p1': (0, 10), 'p2': (0, 10)}


def test_register():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer.space) == 0

    optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    assert len(optimizer.res) == 1
    assert len(optimizer.space) == 1

    optimizer.space.register(params={"p1": 5, "p2": 4}, target=9)
    assert len(optimizer.res) == 2
    assert len(optimizer.space) == 2

    with pytest.raises(KeyError):
        optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    with pytest.raises(KeyError):
        optimizer.register(params={"p1": 5, "p2": 4}, target=9)


def test_probe_lazy():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    optimizer.probe(params={"p1": 1, "p2": 2}, lazy=True)
    assert len(optimizer.space) == 0
    assert len(optimizer._queue) == 1

    optimizer.probe(params={"p1": 6, "p2": 2}, lazy=True)
    assert len(optimizer.space) == 0
    assert len(optimizer._queue) == 2

    optimizer.probe(params={"p1": 6, "p2": 2}, lazy=True)
    assert len(optimizer.space) == 0
    assert len(optimizer._queue) == 3


def test_probe_eager():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    optimizer.probe(params={"p1": 1, "p2": 2}, lazy=False)
    assert len(optimizer.space) == 1
    assert len(optimizer._queue) == 0
    assert optimizer.max["target"] == 3
    assert optimizer.max["params"] == {"p1": 1, "p2": 2}

    optimizer.probe(params={"p1": 3, "p2": 3}, lazy=False)
    assert len(optimizer.space) == 2
    assert len(optimizer._queue) == 0
    assert optimizer.max["target"] == 6
    assert optimizer.max["params"] == {"p1": 3, "p2": 3}

    optimizer.probe(params={"p1": 3, "p2": 3}, lazy=False)
    assert len(optimizer.space) == 2
    assert len(optimizer._queue) == 0
    assert optimizer.max["target"] == 6
    assert optimizer.max["params"] == {"p1": 3, "p2": 3}


def test_suggest_at_random():
    util = UtilityFunction(kind="ucb", kappa=5, xi=0)
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    for _ in range(50):
        sample = optimizer.space.params_to_array(optimizer.suggest(util))
        assert len(sample) == optimizer.space.dim
        assert all(sample >= optimizer.space.bounds[:, 0])
        assert all(sample <= optimizer.space.bounds[:, 1])


def test_suggest_with_one_observation():
    util = UtilityFunction(kind="ucb", kappa=5, xi=0)
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    optimizer.register(params={"p1": 1, "p2": 2}, target=3)

    for _ in range(5):
        sample = optimizer.space.params_to_array(optimizer.suggest(util))
        assert len(sample) == optimizer.space.dim
        assert all(sample >= optimizer.space.bounds[:, 0])
        assert all(sample <= optimizer.space.bounds[:, 1])

    # suggestion = optimizer.suggest(util)
    # for _ in range(5):
    #     new_suggestion = optimizer.suggest(util)
    #     assert suggestion == new_suggestion


def test_prime_queue_all_empty():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer._queue) == 0
    assert len(optimizer.space) == 0

    optimizer._prime_queue(init_points=0)
    assert len(optimizer._queue) == 1
    assert len(optimizer.space) == 0


def test_prime_queue_empty_with_init():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer._queue) == 0
    assert len(optimizer.space) == 0

    optimizer._prime_queue(init_points=5)
    assert len(optimizer._queue) == 5
    assert len(optimizer.space) == 0


def test_prime_queue_with_register():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer._queue) == 0
    assert len(optimizer.space) == 0

    optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    optimizer._prime_queue(init_points=0)
    assert len(optimizer._queue) == 0
    assert len(optimizer.space) == 1


def test_prime_queue_with_register_and_init():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer._queue) == 0
    assert len(optimizer.space) == 0

    optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    optimizer._prime_queue(init_points=3)
    assert len(optimizer._queue) == 3
    assert len(optimizer.space) == 1


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_bayesian_optimization.py
    """
    pytest.main([__file__])
