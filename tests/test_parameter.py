from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import NonlinearConstraint
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

from bayes_opt import BayesianOptimization
from bayes_opt.parameter import CategoricalParameter, FloatParameter, IntParameter, wrap_kernel
from bayes_opt.target_space import TargetSpace


def test_float_parameters():
    def target_func(**kwargs):
        # arbitrary target func
        return sum(kwargs.values())

    pbounds = {"p1": (0, 1), "p2": (1, 2)}
    space = TargetSpace(target_func, pbounds)

    assert space.dim == len(pbounds)
    assert space.empty
    assert space.keys == ["p1", "p2"]

    assert isinstance(space._params_config["p1"], FloatParameter)
    assert isinstance(space._params_config["p2"], FloatParameter)

    assert all(space.bounds[:, 0] == np.array([0, 1]))
    assert all(space.bounds[:, 1] == np.array([1, 2]))
    assert (space.bounds == space.bounds).all()

    point1 = {"p1": 0.2, "p2": 1.5}
    target1 = 1.7
    space.probe(point1)

    point2 = {"p1": 0.5, "p2": 1.0}
    target2 = 1.5
    space.probe(point2)

    assert (space.params[0] == np.fromiter(point1.values(), dtype=float)).all()
    assert (space.params[1] == np.fromiter(point2.values(), dtype=float)).all()

    assert (space.target == np.array([target1, target2])).all()

    p1 = space._params_config["p1"]
    assert p1.to_float(0.2) == 0.2
    assert p1.to_float(np.array(2.3)) == 2.3
    assert p1.to_float(3) == 3.0


def test_int_parameters():
    def target_func(**kwargs):
        assert [isinstance(kwargs[key], int) for key in kwargs]
        # arbitrary target func
        return sum(kwargs.values())

    pbounds = {"p1": (0, 5, int), "p3": (-1, 3, int)}
    space = TargetSpace(target_func, pbounds)

    assert space.dim == len(pbounds)
    assert space.empty
    assert space.keys == ["p1", "p3"]

    assert isinstance(space._params_config["p1"], IntParameter)
    assert isinstance(space._params_config["p3"], IntParameter)

    point1 = {"p1": 2, "p3": 0}
    target1 = 2
    space.probe(point1)

    point2 = {"p1": 1, "p3": -1}
    target2 = 0
    space.probe(point2)

    assert (space.params[0] == np.fromiter(point1.values(), dtype=float)).all()
    assert (space.params[1] == np.fromiter(point2.values(), dtype=float)).all()

    assert (space.target == np.array([target1, target2])).all()

    p1 = space._params_config["p1"]
    assert p1.to_float(0) == 0.0
    assert p1.to_float(np.array(2)) == 2.0
    assert p1.to_float(3) == 3.0

    assert p1.kernel_transform(0) == 0.0
    assert p1.kernel_transform(2.3) == 2.0
    assert p1.kernel_transform(np.array([1.3, 3.6, 7.2])) == pytest.approx(np.array([1, 4, 7]))


def test_cat_parameters():
    fruit_ratings = {"apple": 1.0, "banana": 2.0, "mango": 5.0, "honeydew melon": -10.0, "strawberry": np.pi}

    def target_func(fruit: str):
        return fruit_ratings[fruit]

    fruits = ("apple", "banana", "mango", "honeydew melon", "strawberry")
    pbounds = {"fruit": ("apple", "banana", "mango", "honeydew melon", "strawberry")}
    space = TargetSpace(target_func, pbounds)

    assert space.dim == len(fruits)
    assert space.empty
    assert space.keys == ["fruit"]

    assert isinstance(space._params_config["fruit"], CategoricalParameter)

    assert space.bounds.shape == (len(fruits), 2)
    assert (space.bounds[:, 0] == np.zeros(len(fruits))).all()
    assert (space.bounds[:, 1] == np.ones(len(fruits))).all()

    point1 = {"fruit": "banana"}
    target1 = 2.0
    space.probe(point1)

    point2 = {"fruit": "honeydew melon"}
    target2 = -10.0
    space.probe(point2)

    assert (space.params[0] == np.array([0, 1, 0, 0, 0])).all()
    assert (space.params[1] == np.array([0, 0, 0, 1, 0])).all()

    assert (space.target == np.array([target1, target2])).all()

    p1 = space._params_config["fruit"]
    for i, fruit in enumerate(fruits):
        assert (p1.to_float(fruit) == np.eye(5)[i]).all()

    assert (p1.kernel_transform(np.array([0.8, 0.2, 0.3, 0.5, 0.78])) == np.array([1, 0, 0, 0, 0])).all()
    assert (p1.kernel_transform(np.array([0.78, 0.2, 0.3, 0.5, 0.8])) == np.array([0, 0, 0, 0, 1.0])).all()


def test_cateogrical_valid_bounds():
    pbounds = {"fruit": ("apple", "banana", "mango", "honeydew melon", "banana", "strawberry")}
    with pytest.raises(ValueError):
        TargetSpace(None, pbounds)

    pbounds = {"fruit": ("apple",)}
    with pytest.raises(ValueError):
        TargetSpace(None, pbounds)


def test_to_string():
    pbounds = {"p1": (0, 1), "p2": (1, 2)}
    space = TargetSpace(None, pbounds)

    assert space._params_config["p1"].to_string(0.2, 5) == "0.2  "
    assert space._params_config["p2"].to_string(1.5, 5) == "1.5  "
    assert space._params_config["p1"].to_string(0.2, 3) == "0.2"
    assert space._params_config["p2"].to_string(np.pi, 5) == "3.141"
    assert space._params_config["p1"].to_string(1e-5, 6) == "1e-05 "
    assert space._params_config["p2"].to_string(-1e-5, 6) == "-1e-05"
    assert space._params_config["p1"].to_string(1e-15, 5) == "1e-15"
    assert space._params_config["p1"].to_string(-1.2e-15, 7) == "-1.2..."

    pbounds = {"p1": (0, 5, int), "p3": (-1, 3, int)}
    space = TargetSpace(None, pbounds)

    assert space._params_config["p1"].to_string(2, 5) == "2    "
    assert space._params_config["p3"].to_string(0, 5) == "0    "
    assert space._params_config["p1"].to_string(2, 3) == "2  "
    assert space._params_config["p3"].to_string(-1, 5) == "-1   "
    assert space._params_config["p1"].to_string(123456789, 6) == "123..."

    pbounds = {"fruit": ("apple", "banana", "mango", "honeydew melon", "strawberry")}
    space = TargetSpace(None, pbounds)

    assert space._params_config["fruit"].to_string("apple", 5) == "apple"
    assert space._params_config["fruit"].to_string("banana", 5) == "ba..."
    assert space._params_config["fruit"].to_string("mango", 5) == "mango"
    assert space._params_config["fruit"].to_string("honeydew melon", 10) == "honeyde..."
    assert space._params_config["fruit"].to_string("strawberry", 10) == "strawberry"


def test_preconstructed_parameter():
    pbounds = {"p1": (0, 1), "p2": (1, 2), "p3": IntParameter("p3", (-1, 3))}

    def target_func(p1, p2, p3):
        return p1 + p2 + p3

    optimizer1 = BayesianOptimization(target_func, pbounds)

    pbounds = {"p1": (0, 1), "p2": (1, 2), "p3": (-1, 3, int)}
    optimizer2 = BayesianOptimization(target_func, pbounds)

    assert optimizer1.space.keys == optimizer2.space.keys
    assert (optimizer1.space.bounds == optimizer2.space.bounds).all()
    assert optimizer1.space._params_config["p3"].to_float(2) == 2.0


def test_integration_mixed_optimization():
    fruit_ratings = {"apple": 1.0, "banana": 2.0, "mango": 5.0, "honeydew melon": -10.0, "strawberry": np.pi}

    pbounds = {
        "p1": (0, 1),
        "p2": (1, 2),
        "p3": (-1, 3, int),
        "fruit": ("apple", "banana", "mango", "honeydew melon", "strawberry"),
    }

    def target_func(p1, p2, p3, fruit):
        return p1 + p2 + p3 + fruit_ratings[fruit]

    optimizer = BayesianOptimization(target_func, pbounds)
    optimizer.maximize(init_points=2, n_iter=10)


def test_integration_mixed_optimization_with_constraints():
    fruit_ratings = {"apple": 1.0, "banana": 2.0, "mango": 5.0, "honeydew melon": -10.0, "strawberry": np.pi}

    pbounds = {
        "p1": (0, 1),
        "p2": (1, 2),
        "p3": (-1, 3, int),
        "fruit": ("apple", "banana", "mango", "honeydew melon", "strawberry"),
    }

    def target_func(p1, p2, p3, fruit):
        return p1 + p2 + p3 + fruit_ratings[fruit]

    def constraint_func(p1, p2, p3, fruit):
        return (p1 + p2 + p3 - fruit_ratings[fruit]) ** 2

    constraint = NonlinearConstraint(constraint_func, 0, 4.0)

    optimizer = BayesianOptimization(target_func, pbounds, constraint=constraint)
    init_points = [
        {"p1": 0.5, "p2": 1.5, "p3": 1, "fruit": "banana"},
        {"p1": 0.5, "p2": 1.5, "p3": 2, "fruit": "mango"},
    ]
    for p in init_points:
        optimizer.register(p, target=target_func(**p), constraint_value=constraint_func(**p))
    optimizer.maximize(init_points=0, n_iter=2)


def test_wrapped_kernel_fit():
    pbounds = {"p1": (0, 1), "p2": (1, 10, int)}
    space = TargetSpace(None, pbounds)

    space.register(space.random_sample(0), 1.0)
    space.register(space.random_sample(1), 5.0)

    kernel = wrap_kernel(kernels.Matern(nu=2.5, length_scale=1e5), space.kernel_transform)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5)

    gp.fit(space.params, space.target)

    assert gp.kernel_.length_scale != 1e5


def test_combined_wrapped_kernel_fit():
    pbounds = {"p1": (0, 1), "p2": (1, 10, int)}
    space = TargetSpace(None, pbounds)

    space.register(space.random_sample(0), 1.0)
    space.register(space.random_sample(1), 5.0)

    kernel_fct = kernels.Matern(nu=2.5, length_scale=1e5) + kernels.WhiteKernel(noise_level=1.0)
    kernel = wrap_kernel(kernel_fct, space.kernel_transform)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5)

    gp.fit(space.params, space.target)
