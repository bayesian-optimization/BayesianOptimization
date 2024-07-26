from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import NonlinearConstraint

from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs

test_dir = Path(__file__).parent.resolve()


def test_logs():
    def f(x, y):
        return -(x**2) - (y - 1) ** 2 + 1

    optimizer = BayesianOptimization(f=f, pbounds={"x": (-200, 200), "y": (-200, 200)})
    assert len(optimizer.space) == 0

    load_logs(optimizer, [str(test_dir / "test_logs.log")])
    assert len(optimizer.space) == 5

    load_logs(optimizer, [str(test_dir / "test_logs.log")])
    assert len(optimizer.space) == 5

    other_optimizer = BayesianOptimization(f=lambda x: -(x**2), pbounds={"x": (-2, 2)})
    with pytest.raises(ValueError):
        load_logs(other_optimizer, [str(test_dir / "test_logs.log")])


def test_logs_str():
    def f(x, y):
        return -(x**2) - (y - 1) ** 2 + 1

    optimizer = BayesianOptimization(f=f, pbounds={"x": (-200, 200), "y": (-200, 200)})
    assert len(optimizer.space) == 0

    load_logs(optimizer, str(test_dir / "test_logs.log"))
    assert len(optimizer.space) == 5


def test_logs_bounds():
    def f(x, y):
        return x + y

    optimizer = BayesianOptimization(f=f, pbounds={"x": (-2, 2), "y": (-2, 2)})

    with pytest.warns(UserWarning):
        load_logs(optimizer, [str(test_dir / "test_logs_bounds.log")])

    assert len(optimizer.space) == 5


def test_logs_constraint():
    def f(x, y):
        return -(x**2) - (y - 1) ** 2 + 1

    def c(x, y):
        return x**2 + y**2

    constraint = NonlinearConstraint(c, -np.inf, 3)

    optimizer = BayesianOptimization(f=f, pbounds={"x": (-200, 200), "y": (-200, 200)}, constraint=constraint)

    with pytest.raises(KeyError):
        load_logs(optimizer, [str(test_dir / "test_logs.log")])

    load_logs(optimizer, [str(test_dir / "test_logs_constrained.log")])

    assert len(optimizer.space) == 7


def test_logs_constraint_new_array():
    def f(x, y):
        return -(x**2) - (y - 1) ** 2 + 1

    def c(x, y):
        return np.array(
            [-np.cos(x) * np.cos(y) + np.sin(x) * np.sin(y), -np.cos(x) * np.cos(-y) + np.sin(x) * np.sin(-y)]
        )

    constraint_lower = np.array([-np.inf, -np.inf])
    constraint_upper = np.array([0.6, 0.6])

    constraint = NonlinearConstraint(c, constraint_lower, constraint_upper)

    optimizer = BayesianOptimization(f=f, pbounds={"x": (-200, 200), "y": (-200, 200)}, constraint=constraint)

    with pytest.raises(KeyError):
        load_logs(optimizer, [str(test_dir / "test_logs.log")])

    with pytest.raises(ValueError):
        load_logs(optimizer, [str(test_dir / "test_logs_constrained.log")])

    load_logs(optimizer, [str(test_dir / "test_logs_multiple_constraints.log")])

    print(optimizer.space)
    assert len(optimizer.space) == 12


if __name__ == "__main__":
    r"""
    CommandLine:
        python tests/test_target_space.py
    """
    pytest.main([__file__])
