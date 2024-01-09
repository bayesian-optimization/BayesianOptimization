import pytest
import numpy as np
from bayes_opt.constraint import ConstraintModel
from bayes_opt.target_space import TargetSpace
from bayes_opt.util import NotUniqueError


def target_func(**kwargs):
    # arbitrary target func
    return sum(kwargs.values())


PBOUNDS = {'p1': (0, 1), 'p2': (1, 100)}


def test_keys_and_bounds_in_same_order():
    pbounds = {
        'p1': (0, 1),
        'p3': (0, 3),
        'p2': (0, 2),
        'p4': (0, 4),
    }
    space = TargetSpace(target_func, pbounds)

    assert space.dim == len(pbounds)
    assert space.empty
    assert space.keys == ["p1", "p2",  "p3",  "p4"]
    assert all(space.bounds[:, 0] == np.array([0, 0, 0, 0]))
    assert all(space.bounds[:, 1] == np.array([1, 2, 3, 4]))


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


def test_as_array():
    space = TargetSpace(target_func, PBOUNDS)

    x = space._as_array([0, 1])
    assert x.shape == (2,)
    assert all(x == np.array([0, 1]))

    x = space._as_array({"p2": 1, "p1": 2})
    assert x.shape == (2,)
    assert all(x == np.array([2, 1]))

    with pytest.raises(ValueError):
        x = space._as_array([2, 1, 7])
    with pytest.raises(ValueError):
        x = space._as_array({"p2": 1, "p1": 2, "other": 7})
    with pytest.raises(ValueError):
        x = space._as_array({"p2": 1})
    with pytest.raises(ValueError):
        x = space._as_array({"other": 7})


def test_register():
    PBOUNDS = {'p1': (0, 10), 'p2': (1, 100)}
    space = TargetSpace(target_func, PBOUNDS)

    assert len(space) == 0
    # registering with dict
    space.register(params={"p1": 1, "p2": 2}, target=3)
    assert len(space) == 1
    assert all(space.params[0] == np.array([1, 2]))
    assert all(space.target == np.array([3]))

    # registering with array
    space.register(params={"p1": 5, "p2": 4}, target=9)
    assert len(space) == 2
    assert all(space.params[1] == np.array([5, 4]))
    assert all(space.target == np.array([3, 9]))

    with pytest.raises(NotUniqueError):
        space.register(params={"p1": 1, "p2": 2}, target=3)
    with pytest.raises(NotUniqueError):
        space.register(params={"p1": 5, "p2": 4}, target=9)


def test_register_with_constraint():
    PBOUNDS = {'p1': (0, 10), 'p2': (1, 100)}
    constraint = ConstraintModel(lambda x: x, -2, 2)
    space = TargetSpace(target_func, PBOUNDS, constraint=constraint)

    assert len(space) == 0
    # registering with dict
    space.register(params={"p1": 1, "p2": 2}, target=3, constraint_value=0.)
    assert len(space) == 1
    assert all(space.params[0] == np.array([1, 2]))
    assert all(space.target == np.array([3]))
    assert all(space.constraint_values == np.array([0]))

    # registering with array
    space.register(params={"p1": 5, "p2": 4}, target=9, constraint_value=2)
    assert len(space) == 2
    assert all(space.params[1] == np.array([5, 4]))
    assert all(space.target == np.array([3, 9]))
    assert all(space.constraint_values == np.array([0, 2]))

    with pytest.raises(ValueError):
        space.register(params={"p1": 2, "p2": 2}, target=3)


def test_register_point_beyond_bounds():
    PBOUNDS = {'p1': (0, 1), 'p2': (1, 10)}
    space = TargetSpace(target_func, PBOUNDS)

    with pytest.warns(UserWarning):
        space.register(params={"p1": 0.5, "p2": 20}, target=2.5)


def test_probe():
    PBOUNDS = {'p1': (0, 10), 'p2': (1, 100)}
    space = TargetSpace(target_func, PBOUNDS, allow_duplicate_points=True)

    assert len(space) == 0
    # probing with dict
    space.probe(params={"p1": 1, "p2": 2})
    assert len(space) == 1
    assert all(space.params[0] == np.array([1, 2]))
    assert all(space.target == np.array([3]))

    # probing with array
    space.probe(np.array([5, 4]))
    assert len(space) == 2
    assert all(space.params[1] == np.array([5, 4]))
    assert all(space.target == np.array([3, 9]))

    # probing same point with dict
    space.probe(params={"p1": 1, "p2": 2})
    assert len(space) == 3
    assert all(space.params[1] == np.array([5, 4]))
    assert all(space.target == np.array([3, 9, 3]))

    # probing same point with array
    space.probe(np.array([5, 4]))
    assert len(space) == 4
    assert all(space.params[1] == np.array([5, 4]))
    assert all(space.target == np.array([3, 9, 3 , 9]))


def test_random_sample():
    pbounds = {
        'p1': (0, 1),
        'p3': (0, 3),
        'p2': (0, 2),
        'p4': (0, 4),
    }
    space = TargetSpace(target_func, pbounds, random_state=8)

    for _ in range(50):
        random_sample = space.random_sample()
        assert len(random_sample) == space.dim
        assert all(random_sample >= space.bounds[:, 0])
        assert all(random_sample <= space.bounds[:, 1])


def test_y_max():
    space = TargetSpace(target_func, PBOUNDS)
    assert space._target_max() == None
    space.probe(params={"p1": 1, "p2": 7})
    space.probe(params={"p1": 0.5, "p2": 1})
    space.probe(params={"p1": 0, "p2": 1})
    assert space._target_max() == 8


def test_y_max_with_constraint():
    PBOUNDS = {'p1': (0, 10), 'p2': (1, 100)}
    constraint = ConstraintModel(lambda p1, p2: p1-p2, -2, 2)
    space = TargetSpace(target_func, PBOUNDS, constraint)
    assert space._target_max() == None
    space.probe(params={"p1": 1, "p2": 2}) # Feasible
    space.probe(params={"p1": 5, "p2": 1}) # Unfeasible
    space.probe(params={"p1": 0, "p2": 1}) # Feasible
    assert space._target_max() == 3


def test_y_max_within_pbounds():
    PBOUNDS = {'p1': (0, 2), 'p2': (1, 100)}
    space = TargetSpace(target_func, PBOUNDS)
    assert space._target_max() == None
    space.probe(params={"p1": 1, "p2": 2})
    space.probe(params={"p1": 0, "p2": 1})
    with pytest.warns(UserWarning):
        space.probe(params={"p1": 5, "p2": 1})
    assert space._target_max() == 3


def test_max():
    PBOUNDS = {'p1': (0, 10), 'p2': (1, 100)}
    space = TargetSpace(target_func, PBOUNDS)
    
    assert space.max() == None
    space.probe(params={"p1": 1, "p2": 2})
    space.probe(params={"p1": 5, "p2": 4})
    space.probe(params={"p1": 2, "p2": 3})
    space.probe(params={"p1": 1, "p2": 6})
    assert space.max() == {"params": {"p1": 5, "p2": 4}, "target": 9}


def test_max_with_constraint():
    PBOUNDS = {'p1': (0, 10), 'p2': (1, 100)}
    constraint = ConstraintModel(lambda p1, p2: p1-p2, -2, 2)
    space = TargetSpace(target_func, PBOUNDS, constraint=constraint)

    assert space.max() == None
    space.probe(params={"p1": 1, "p2": 2}) # Feasible
    space.probe(params={"p1": 5, "p2": 8}) # Unfeasible
    space.probe(params={"p1": 2, "p2": 3}) # Feasible
    space.probe(params={"p1": 1, "p2": 6}) # Unfeasible
    assert space.max() == {"params": {"p1": 2, "p2": 3}, "target": 5, "constraint": -1}


def test_max_with_constraint_identical_target_value():
    PBOUNDS = {'p1': (0, 10), 'p2': (1, 100)}
    constraint = ConstraintModel(lambda p1, p2: p1-p2, -2, 2)
    space = TargetSpace(target_func, PBOUNDS, constraint=constraint)

    assert space.max() == None
    space.probe(params={"p1": 1, "p2": 2}) # Feasible
    space.probe(params={"p1": 0, "p2": 5}) # Unfeasible, target value is 5, should not be selected
    space.probe(params={"p1": 5, "p2": 8}) # Unfeasible
    space.probe(params={"p1": 2, "p2": 3}) # Feasible, target value is also 5
    space.probe(params={"p1": 1, "p2": 6}) # Unfeasible
    assert space.max() == {"params": {"p1": 2, "p2": 3}, "target": 5, "constraint": -1}


def test_res():
    PBOUNDS = {'p1': (0, 10), 'p2': (1, 100)}
    space = TargetSpace(target_func, PBOUNDS)

    assert space.res() == []
    space.probe(params={"p1": 1, "p2": 2})
    space.probe(params={"p1": 5, "p2": 4})
    space.probe(params={"p1": 2, "p2": 3})
    space.probe(params={"p1": 1, "p2": 6})

    expected_res = [
        {"params":  {"p1": 1, "p2": 2}, "target": 3},
        {"params":  {"p1": 5, "p2": 4}, "target": 9},
        {"params":  {"p1": 2, "p2": 3}, "target": 5},
        {"params":  {"p1": 1, "p2": 6}, "target": 7},
    ]
    assert len(space.res()) == 4
    assert space.res() == expected_res


def test_set_bounds():
    pbounds = {
        'p1': (0, 1),
        'p3': (0, 3),
        'p2': (0, 2),
        'p4': (0, 4),
    }
    space = TargetSpace(target_func, pbounds)

    # Ignore unknown keys
    space.set_bounds({"other": (7, 8)})
    assert all(space.bounds[:, 0] == np.array([0, 0, 0, 0]))
    assert all(space.bounds[:, 1] == np.array([1, 2, 3, 4]))

    # Update bounds accordingly
    space.set_bounds({"p2": (1, 8)})
    assert all(space.bounds[:, 0] == np.array([0, 1, 0, 0]))
    assert all(space.bounds[:, 1] == np.array([1, 8, 3, 4]))


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_target_space.py
    """
    pytest.main([__file__])
