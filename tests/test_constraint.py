from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import NonlinearConstraint

from bayes_opt import BayesianOptimization, ConstraintModel


@pytest.fixture
def target_function():
    return lambda x, y: np.cos(2 * x) * np.cos(y) + np.sin(x)


@pytest.fixture
def constraint_function():
    return lambda x, y: np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y)


def test_constraint_property(target_function, constraint_function):
    constraint_limit_upper = 0.5
    constraint = NonlinearConstraint(constraint_function, -np.inf, constraint_limit_upper)
    pbounds = {"x": (0, 6), "y": (0, 6)}

    optimizer = BayesianOptimization(
        f=target_function, constraint=constraint, pbounds=pbounds, verbose=0, random_state=1
    )
    assert isinstance(optimizer.constraint, ConstraintModel)
    assert isinstance(optimizer.space.constraint, ConstraintModel)


def test_single_constraint_upper(target_function, constraint_function):
    constraint_limit_upper = 0.5

    constraint = NonlinearConstraint(constraint_function, -np.inf, constraint_limit_upper)
    pbounds = {"x": (0, 6), "y": (0, 6)}

    optimizer = BayesianOptimization(
        f=target_function, constraint=constraint, pbounds=pbounds, verbose=0, random_state=1
    )

    optimizer.maximize(init_points=2, n_iter=10)

    assert constraint_function(**optimizer.max["params"]) <= constraint_limit_upper


def test_single_constraint_lower(target_function, constraint_function):
    constraint_limit_lower = -0.5

    constraint = NonlinearConstraint(constraint_function, constraint_limit_lower, np.inf)
    pbounds = {"x": (0, 6), "y": (0, 6)}

    optimizer = BayesianOptimization(
        f=target_function, constraint=constraint, pbounds=pbounds, verbose=0, random_state=1
    )

    optimizer.maximize(init_points=2, n_iter=10)

    assert constraint_function(**optimizer.max["params"]) >= constraint_limit_lower


def test_single_constraint_lower_upper(target_function, constraint_function):
    constraint_limit_lower = -0.5
    constraint_limit_upper = 0.5

    constraint = NonlinearConstraint(constraint_function, constraint_limit_lower, constraint_limit_upper)
    pbounds = {"x": (0, 6), "y": (0, 6)}

    optimizer = BayesianOptimization(
        f=target_function, constraint=constraint, pbounds=pbounds, verbose=0, random_state=1
    )

    assert optimizer.constraint.lb == constraint.lb
    assert optimizer.constraint.ub == constraint.ub

    optimizer.maximize(init_points=2, n_iter=10)

    # Check limits
    assert constraint_function(**optimizer.max["params"]) <= constraint_limit_upper
    assert constraint_function(**optimizer.max["params"]) >= constraint_limit_lower

    # Exclude the last sampled point, because the constraint is not fitted on that.
    res = np.array(
        [[r["target"], r["constraint"], r["params"]["x"], r["params"]["y"]] for r in optimizer.res[:-1]]
    )

    xy = res[:, [2, 3]]
    x = res[:, 2]
    y = res[:, 3]

    # Check accuracy of approximation for sampled points
    assert constraint_function(x, y) == pytest.approx(optimizer.constraint.approx(xy), rel=1e-4, abs=1e-4)


def test_multiple_constraints(target_function):
    def constraint_function_2_dim(x, y):
        return np.array(
            [-np.cos(x) * np.cos(y) + np.sin(x) * np.sin(y), -np.cos(x) * np.cos(-y) + np.sin(x) * np.sin(-y)]
        )

    constraint_limit_lower = np.array([-np.inf, -np.inf])
    constraint_limit_upper = np.array([0.6, 0.6])

    conmod = NonlinearConstraint(constraint_function_2_dim, constraint_limit_lower, constraint_limit_upper)
    pbounds = {"x": (0, 6), "y": (0, 6)}

    optimizer = BayesianOptimization(
        f=target_function, constraint=conmod, pbounds=pbounds, verbose=0, random_state=1
    )

    optimizer.maximize(init_points=2, n_iter=10)

    constraint_at_max = constraint_function_2_dim(**optimizer.max["params"])
    assert np.all(
        (constraint_at_max <= constraint_limit_upper) & (constraint_at_max >= constraint_limit_lower)
    )

    params = optimizer.res[0]["params"]
    x, y = params["x"], params["y"]

    assert constraint_function_2_dim(x, y) == pytest.approx(
        optimizer.constraint.approx(np.array([x, y])), rel=1e-3, abs=1e-3
    )


def test_kwargs_not_the_same(target_function):
    def target_function(x, y):
        return np.cos(2 * x) * np.cos(y) + np.sin(x)

    def constraint_function(a, b):
        return np.cos(a) * np.cos(b) - np.sin(a) * np.sin(b)

    constraint_limit_upper = 0.5

    constraint = NonlinearConstraint(constraint_function, -np.inf, constraint_limit_upper)
    pbounds = {"x": (0, 6), "y": (0, 6)}

    optimizer = BayesianOptimization(
        f=target_function, constraint=constraint, pbounds=pbounds, verbose=0, random_state=1
    )
    with pytest.raises(TypeError, match="Encountered TypeError when evaluating"):
        optimizer.maximize(init_points=2, n_iter=10)


def test_lower_less_than_upper(target_function):
    def target_function(x, y):
        return np.cos(2 * x) * np.cos(y) + np.sin(x)

    def constraint_function_2_dim(x, y):
        return np.array(
            [-np.cos(x) * np.cos(y) + np.sin(x) * np.sin(y), -np.cos(x) * np.cos(-y) + np.sin(x) * np.sin(-y)]
        )

    constraint_limit_lower = np.array([0.6, -np.inf])
    constraint_limit_upper = np.array([0.3, 0.6])

    conmod = NonlinearConstraint(constraint_function_2_dim, constraint_limit_lower, constraint_limit_upper)
    pbounds = {"x": (0, 6), "y": (0, 6)}

    with pytest.raises(ValueError):
        BayesianOptimization(f=target_function, constraint=conmod, pbounds=pbounds, verbose=0, random_state=1)


def test_null_constraint_function():
    constraint = ConstraintModel(None, np.array([0, 0]), np.array([1, 1]))
    with pytest.raises(ValueError, match="No constraint function was provided."):
        constraint.eval()
