from bayes_opt.target_space import TargetSpace
import pytest
import numpy as np


def target_func(**kw):
    # arbitrary target func
    return sum(kw.values())


def test_empty():
    pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    space = TargetSpace(target_func, pbounds)
    space._assert_internal_invariants(fast=False)
    assert space._n_alloc_rows == len(space)


def test_one_point():
    pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    space = TargetSpace(target_func, pbounds)
    x = space.random_points(1)[0]
    space.observe_point(x)
    space._assert_internal_invariants(fast=False)
    assert space._n_alloc_rows > len(space)


def test_two_points():
    """
    pytest tests/test_target_space.py::test_two_points
    """
    pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    space = TargetSpace(target_func, pbounds)
    for x in space.random_points(2):
        space.observe_point(x)
        space._assert_internal_invariants(fast=False)
    space._assert_internal_invariants(fast=False)
    assert space._n_alloc_rows == len(space)


def test_nonunique_add():
    # Adding non-unique values throws a KeyError
    pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    space = TargetSpace(target_func, pbounds)
    x = [0, 0]
    y = 0
    space.add_observation(x, y)

    with pytest.raises(KeyError):
        space.add_observation(x, y)

    space._assert_internal_invariants(fast=False)


def test_nonunique_observe():
    # Simply re-observing a non-unique values returns the cached result
    pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    space = TargetSpace(target_func, pbounds)
    x = [0, 0]
    y = 0
    space.add_observation(x, y)

    with pytest.raises(KeyError):
        space.add_observation(x, y)

    space._assert_internal_invariants(fast=False)


def test_contains():
    # Simply re-observing a non-unique values returns the cached result
    pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    space = TargetSpace(target_func, pbounds)
    # add 1000 random points
    for x in space.random_points(1000):
        space.observe_point(x)
    # now all points should be unique, so test contains
    space2 = TargetSpace(target_func, pbounds)
    for x in space.X:
        assert x not in space2
        y = space2.observe_point(x)
        assert x in space2
        assert y == space2.observe_point(x)
    space2._assert_internal_invariants(fast=False)


@pytest.mark.parametrize("m", [0, 1, 2, 5, 20, 100])
@pytest.mark.parametrize("n", [0, 1, 2, 3, 10])
def test_m_random_nd_points(m, n):
    pbounds = {'p{}'.format(i): (0, i) for i in range(n)}
    space = TargetSpace(target_func, pbounds, random_state=0)

    X = space.random_points(m)
    assert X.shape[0] == m
    assert X.shape[1] == n

    for i in range(n):
        lower, upper = pbounds[space.keys[i]]
        assert np.all(X.T[i] >= lower)
        assert np.all(X.T[i] <= upper)


@pytest.mark.parametrize("m", [0, 1, 2, 5, 20, 100])
@pytest.mark.parametrize("n", [0, 1, 2, 3, 10])
def test_observe_m_nd_points(m, n):
    pbounds = {'p{}'.format(i): (0, i) for i in range(n)}
    space = TargetSpace(target_func, pbounds)
    for x in space.random_points(m):
        space.observe_point(x)
        space._assert_internal_invariants(fast=False)
    space._assert_internal_invariants(fast=False)


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_target_space.py
    """
    import pytest
    pytest.main([__file__])
