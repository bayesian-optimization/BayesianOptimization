import numpy as np
from bayes_opt.target_space import TargetSpace
from bayes_opt.parameter import FloatParameter, IntParameter, CategoricalParameter
import pytest

def test_float_parameters():
    def target_func(**kwargs):
        # arbitrary target func
        return sum(kwargs.values())
    pbounds = {
        'p1': (0, 1),
        'p2': (1, 2)
    }
    space = TargetSpace(target_func, pbounds)

    assert space.dim == len(pbounds)
    assert space.empty
    assert space.keys == ["p1", "p2"]

    assert isinstance(space._params_config['p1'], FloatParameter)
    assert isinstance(space._params_config['p2'], FloatParameter)

    assert all(space.float_bounds[:, 0] == np.array([0, 1]))
    assert all(space.float_bounds[:, 1] == np.array([1, 2]))
    assert (space.bounds == space.float_bounds).all()

    point1 = {'p1': 0.2, 'p2': 1.5}
    target1 = 1.7
    space.probe(point1)

    point2 = {'p1': 0.2, 'p2': 2.5}
    target2 = 2.7
    space.probe(point2)

    assert (space.params[0] == np.fromiter(point1.values(), dtype=float)).all()
    assert (space.params[1] == np.fromiter(point2.values(), dtype=float)).all()

    assert (space.target == np.array([target1, target2])).all()


def test_int_parameters():
    def target_func(**kwargs):
        assert [isinstance(kwargs[key], int) for key in kwargs]
        # arbitrary target func
        return sum(kwargs.values())

    pbounds = {
        'p1': (0, 5, int),
        'p3': (-1, 3, int)
    }
    space = TargetSpace(target_func, pbounds)

    assert space.dim == len(pbounds)
    assert space.empty
    assert space.keys == ["p1", "p3"]

    assert isinstance(space._params_config['p1'], IntParameter)
    assert isinstance(space._params_config['p3'], IntParameter)

    assert pytest.approx(space.float_bounds[:, 0], abs=0.001) == np.array([-0.5, -1.5]) # sub 0.5
    assert pytest.approx(space.float_bounds[:, 1], abs=0.001) == np.array([5.5, 3.5]) # add 0.5
    assert (space.bounds != space.float_bounds).any() # bounds are modified from float bounds

    point1 = {'p1': 2, 'p3': 0}
    target1 = 2
    space.probe(point1)

    point2 = {'p1': 1, 'p3': -1}
    target2 = 0
    space.probe(point2)

    assert (space.params[0] == np.fromiter(point1.values(), dtype=float)).all()
    assert (space.params[1] == np.fromiter(point2.values(), dtype=float)).all()

    assert (space.target == np.array([target1, target2])).all()


def test_cat_parameters():

    fruit_ratings = {
        'apple': 1.,
        'banana': 2.,
        'mango': 5.,
        'honeydew melon': -10.,
        'straberry': np.pi
    }

    def target_func(fruit: str):
        return fruit_ratings[fruit]
    
    fruits = ('apple', 'banana', 'mango', 'honeydew melon', 'straberry')
    pbounds = {
        'fruit': ('apple', 'banana', 'mango', 'honeydew melon', 'straberry'),
    }
    space = TargetSpace(target_func, pbounds)

    assert space.dim == len(fruits)
    assert space.empty
    assert space.keys == ["fruit"]

    assert isinstance(space._params_config['fruit'], CategoricalParameter)

    assert space.float_bounds.shape == (len(fruits), 2)
    assert (space.float_bounds[:, 0] == np.zeros(len(fruits))).all()
    assert (space.float_bounds[:, 1] == np.ones(len(fruits))).all()

    point1 = {'fruit': 'banana'}
    target1 = 2.
    space.probe(point1)

    point2 = {'fruit': 'honeydew melon'}
    target2 = -10.
    space.probe(point2)

    assert (space.params[0] == np.array([0, 1, 0, 0, 0])).all()
    assert (space.params[1] == np.array([0, 0, 0, 1, 0])).all()

    assert (space.target == np.array([target1, target2])).all()
