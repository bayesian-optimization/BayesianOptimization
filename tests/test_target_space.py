from __future__ import annotations

import numpy as np
import pytest

from bayes_opt.constraint import ConstraintModel
from bayes_opt.exception import NotUniqueError
from bayes_opt.target_space import TargetSpace


def target_func(**kwargs):
    # arbitrary target func
    return sum(kwargs.values())


PBOUNDS = {"p1": (0, 1), "p2": (1, 100)}


def test_keys_and_bounds_in_same_order():
    pbounds = {"p1": (0, 1), "p3": (0, 3), "p2": (0, 2), "p4": (0, 4)}
    space = TargetSpace(target_func, pbounds)

    assert space.dim == len(pbounds)
    assert space.empty
    assert space.keys == ["p1", "p3", "p2", "p4"]
    assert all(space.bounds[:, 0] == np.array([0, 0, 0, 0]))
    assert all(space.bounds[:, 1] == np.array([1, 3, 2, 4]))


def test_params_to_array():
    space = TargetSpace(target_func, PBOUNDS)

    assert all(space.params_to_array({"p1": 2, "p2": 3}) == np.array([2, 3]))
    assert all(space.params_to_array({"p2": 2, "p1": 9}) == np.array([9, 2]))
    with pytest.raises(ValueError):
        space.params_to_array({"p2": 1})
    with pytest.raises(ValueError):
        space.params_to_array({"p2": 1, "p1": 7, "other": 4})
    with pytest.raises(ValueError):
        space.params_to_array({"other": 1})


def test_array_to_params():
    space = TargetSpace(target_func, PBOUNDS)

    assert space.array_to_params(np.array([2, 3])) == {"p1": 2, "p2": 3}
    with pytest.raises(ValueError):
        space.array_to_params(np.array([2]))
    with pytest.raises(ValueError):
        space.array_to_params(np.array([2, 3, 5]))


def test_to_float():
    space = TargetSpace(target_func, PBOUNDS)

    x = space._to_float({"p2": 0, "p1": 1})
    assert x.shape == (2,)
    assert all(x == np.array([1, 0]))

    with pytest.raises(ValueError):
        x = space._to_float([0, 1])
    with pytest.raises(ValueError):
        x = space._to_float([2, 1, 7])
    with pytest.raises(ValueError):
        x = space._to_float({"p2": 1, "p1": 2, "other": 7})
    with pytest.raises(ValueError):
        x = space._to_float({"p2": 1})
    with pytest.raises(ValueError):
        x = space._to_float({"other": 7})


def test_register():
    PBOUNDS = {"p1": (0, 10), "p2": (1, 100)}
    space = TargetSpace(target_func, PBOUNDS)

    assert len(space) == 0
    # registering with dict
    space.register(params={"p1": 1, "p2": 2}, target=3)
    assert len(space) == 1
    assert all(space.params[0] == np.array([1, 2]))
    assert all(space.target == np.array([3]))

    # registering with dict out of order
    space.register(params={"p2": 4, "p1": 5}, target=9)
    assert len(space) == 2
    assert all(space.params[1] == np.array([5, 4]))
    assert all(space.target == np.array([3, 9]))

    # registering with array
    space.register(params=np.array([0, 1]), target=1)
    assert len(space) == 3
    assert all(space.params[2] == np.array([0, 1]))
    assert all(space.target == np.array([3, 9, 1]))

    with pytest.raises(NotUniqueError):
        space.register(params={"p1": 1, "p2": 2}, target=3)
    with pytest.raises(NotUniqueError):
        space.register(params={"p1": 5, "p2": 4}, target=9)


def test_register_with_constraint():
    """
    Tests that registering points with a constraint in TargetSpace requires a constraint value,
    correctly stores constraint values, and raises a ValueError if the constraint value is missing.
    """
    constraint = ConstraintModel(lambda x: x, -2, 2, transform=None)
    space = TargetSpace(target_func, PBOUNDS)
    space.set_constraint(constraint)

    assert len(space) == 0
    # registering with dict
    space.register(params={"p1": 1, "p2": 2}, target=3, constraint_value=0.0)
    assert len(space) == 1
    assert all(space.params[0] == np.array([1, 2]))
    assert all(space.target == np.array([3]))
    assert all(space.constraint_values == np.array([0]))

    # registering with array
    space.register(params={"p1": 0.5, "p2": 4}, target=4.5, constraint_value=2)
    assert len(space) == 2
    assert all(space.params[1] == np.array([0.5, 4]))
    assert all(space.target == np.array([3, 4.5]))
    assert all(space.constraint_values == np.array([0, 2]))

    with pytest.raises(ValueError):
        space.register(params={"p1": 0.2, "p2": 2}, target=2.2)


def test_register_point_beyond_bounds():
    PBOUNDS = {"p1": (0, 1), "p2": (1, 10)}
    space = TargetSpace(target_func, PBOUNDS)

    with pytest.warns(UserWarning):
        space.register(params={"p1": 0.5, "p2": 20}, target=2.5)


def test_probe():
    PBOUNDS = {"p1": (0, 10), "p2": (1, 100)}
    space = TargetSpace(target_func, PBOUNDS, allow_duplicate_points=True)

    assert len(space) == 0
    # probing with dict
    space.probe(params={"p1": 1, "p2": 2})
    assert len(space) == 1
    assert all(space.params[-1] == np.array([1, 2]))
    assert all(space.target == np.array([3]))

    # probing with array
    space.probe(np.array([5, 4]))
    assert len(space) == 2
    assert all(space.params[1] == np.array([5, 4]))
    assert all(space.target == np.array([3, 9]))

    # probing same point with dict
    space.probe(params={"p1": 1, "p2": 2})
    assert len(space) == 3
    assert all(space.params[2] == np.array([1, 2]))
    assert all(space.target == np.array([3, 9, 3]))

    # probing same point with array
    space.probe(np.array([5, 4]))
    assert len(space) == 4
    assert all(space.params[1] == np.array([5, 4]))
    assert all(space.target == np.array([3, 9, 3, 9]))

    space = TargetSpace(target_func, PBOUNDS, allow_duplicate_points=False)

    # register wrong target to check probe doesn't recompute a duplicate point
    space.register(params={"p1": 1, "p2": 2}, target=5)

    # probing same point with dict
    target_ = space.probe(params={"p1": 1, "p2": 2})
    assert target_ == 5

    # probing same point with array
    target_ = space.probe(np.array([1, 2]))
    assert target_ == 5


def test_random_sample():
    pbounds = {"p1": (0, 1), "p3": (0, 3), "p2": (0, 2), "p4": (0, 4)}
    space = TargetSpace(target_func, pbounds, random_state=8)

    for _ in range(50):
        random_sample = space.random_sample()
        assert len(random_sample) == space.dim
        assert all(random_sample >= space.bounds[:, 0])
        assert all(random_sample <= space.bounds[:, 1])


def test_y_max():
    space = TargetSpace(target_func, PBOUNDS)
    assert space._target_max() is None
    space.probe(params={"p1": 1, "p2": 7})
    space.probe(params={"p1": 0.5, "p2": 1})
    space.probe(params={"p1": 0, "p2": 1})
    assert space._target_max() == 8


def test_y_max_with_constraint():
    """
    Tests that `_target_max` returns the maximum target value among feasible points when a constraint is set.
    
    Verifies that only points satisfying the constraint are considered, and that the method returns `None` if no feasible points exist.
    """
    PBOUNDS = {"p1": (0, 10), "p2": (1, 100)}
    constraint = ConstraintModel(lambda p1, p2: p1 - p2, -2, 2)
    space = TargetSpace(target_func, PBOUNDS)
    space.set_constraint(constraint)
    assert space._target_max() is None
    space.probe(params={"p1": 1, "p2": 2})  # Feasible
    space.probe(params={"p1": 5, "p2": 1})  # Unfeasible
    space.probe(params={"p1": 0, "p2": 1})  # Feasible
    assert space._target_max() == 3


def test_y_max_within_pbounds():
    PBOUNDS = {"p1": (0, 2), "p2": (1, 100)}
    space = TargetSpace(target_func, PBOUNDS)
    assert space._target_max() is None
    space.probe(params={"p1": 1, "p2": 2})
    space.probe(params={"p1": 0, "p2": 1})
    with pytest.warns(UserWarning):
        space.probe(params={"p1": 5, "p2": 1})
    assert space._target_max() == 3


def test_max():
    PBOUNDS = {"p1": (0, 10), "p2": (1, 100)}
    space = TargetSpace(target_func, PBOUNDS)

    assert space.max() is None
    space.probe(params={"p1": 1, "p2": 2})
    space.probe(params={"p1": 5, "p2": 4})
    space.probe(params={"p1": 2, "p2": 3})
    space.probe(params={"p1": 1, "p2": 6})
    assert space.max() == {"params": {"p1": 5, "p2": 4}, "target": 9}


def test_max_with_constraint():
    """
    Tests that the `max` method of `TargetSpace` returns the best feasible point when a constraint is set.
    
    Verifies that only points satisfying the constraint are considered, and the returned result includes the parameters, target value, and constraint value.
    """
    PBOUNDS = {"p1": (0, 10), "p2": (1, 100)}
    constraint = ConstraintModel(lambda p1, p2: p1 - p2, -2, 2)
    space = TargetSpace(target_func, PBOUNDS)
    space.set_constraint(constraint)

    assert space.max() is None
    space.probe(params={"p1": 1, "p2": 2})  # Feasible
    space.probe(params={"p1": 5, "p2": 8})  # Unfeasible
    space.probe(params={"p1": 2, "p2": 3})  # Feasible
    space.probe(params={"p1": 1, "p2": 6})  # Unfeasible
    assert space.max() == {"params": {"p1": 2, "p2": 3}, "target": 5, "constraint": -1}


def test_max_with_constraint_identical_target_value():
    """
    Tests that `TargetSpace.max()` returns the best feasible point when multiple points have identical target values but different constraint satisfaction.
    
    Ensures that only points satisfying the constraint are considered, and among feasible points with the same target value, the correct one is selected.
    """
    PBOUNDS = {"p1": (0, 10), "p2": (1, 100)}
    constraint = ConstraintModel(lambda p1, p2: p1 - p2, -2, 2)
    space = TargetSpace(target_func, PBOUNDS)
    space.set_constraint(constraint)

    assert space.max() is None
    space.probe(params={"p1": 1, "p2": 2})  # Feasible
    space.probe(params={"p1": 0, "p2": 5})  # Unfeasible, target value is 5, should not be selected
    space.probe(params={"p1": 5, "p2": 8})  # Unfeasible
    space.probe(params={"p1": 2, "p2": 3})  # Feasible, target value is also 5
    space.probe(params={"p1": 1, "p2": 6})  # Unfeasible
    assert space.max() == {"params": {"p1": 2, "p2": 3}, "target": 5, "constraint": -1}


def test_res():
    PBOUNDS = {"p1": (0, 10), "p2": (1, 100)}
    space = TargetSpace(target_func, PBOUNDS)

    assert space.res() == []
    space.probe(params={"p1": 1, "p2": 2})
    space.probe(params={"p1": 5, "p2": 4})
    space.probe(params={"p1": 2, "p2": 3})
    space.probe(params={"p1": 1, "p2": 6})

    expected_res = [
        {"params": {"p1": 1, "p2": 2}, "target": 3},
        {"params": {"p1": 5, "p2": 4}, "target": 9},
        {"params": {"p1": 2, "p2": 3}, "target": 5},
        {"params": {"p1": 1, "p2": 6}, "target": 7},
    ]
    assert len(space.res()) == 4
    assert space.res() == expected_res


def test_set_bounds():
    pbounds = {"p1": (0, 1), "p3": (0, 3), "p2": (0, 2), "p4": (0, 4)}
    space = TargetSpace(target_func, pbounds)

    # Ignore unknown keys
    space.set_bounds({"other": (7, 8)})
    assert all(space.bounds[:, 0] == np.array([0, 0, 0, 0]))
    assert all(space.bounds[:, 1] == np.array([1, 3, 2, 4]))

    # Update bounds accordingly
    space.set_bounds({"p2": (1, 8)})
    assert all(space.bounds[:, 0] == np.array([0, 0, 1, 0]))
    assert all(space.bounds[:, 1] == np.array([1, 3, 8, 4]))


def test_no_target_func():
    target_space = TargetSpace(None, PBOUNDS)
    with pytest.raises(ValueError, match="No target function has been provided."):
        target_space.probe({"p1": 1, "p2": 2})


def test_change_typed_bounds():
    pbounds = {
        "p1": (0, 1),
        "p2": (1, 2),
        "p3": (-1, 3, int),
        "fruit": ("apple", "banana", "mango", "honeydew melon", "strawberry"),
    }

    space = TargetSpace(None, pbounds)

    with pytest.raises(ValueError):
        space.set_bounds({"fruit": ("apple", "banana", "mango", "honeydew melon")})

    with pytest.raises(ValueError):
        space.set_bounds({"p3": (-1, 2, float)})
