from bayes_opt.bayesian_optimization import BayesianOptimization
import pytest


def test_register_dummy():
    optimizer = BayesianOptimization(f=None, pbounds={"x": (-1, 1)}, random_state=42)

    optimizer.register_dummy({"x": 0}, default_value=0)
    optimizer.register({"x": 0}, 1)
    with pytest.raises(KeyError):
        optimizer.register({"x": 0}, 1)

def test_res_with_dummy():
    optimizer = BayesianOptimization(f=None, pbounds={"x": (-1, 1)}, random_state=42)

    optimizer.register_dummy({"x": 0}, default_value=0)
    optimizer.register({"x": -1}, -1)
    assert len(optimizer.res) == 1


def test_max_with_dummy():
    optimizer = BayesianOptimization(f=None, pbounds={"x": (-1, 1)}, random_state=42)

    optimizer.register_dummy({"x": 0}, default_value=0)
    optimizer.register({"x": -1}, -1)
    best = optimizer.max
    assert best["params"]["x"] == -1
    assert best["target"] == -1
