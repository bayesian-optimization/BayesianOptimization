import pytest
import numpy as np
from bayes_opt import acquisition
from sklearn.gaussian_process import GaussianProcessRegressor
from bayes_opt.target_space import TargetSpace
from bayes_opt.constraint import ConstraintModel
from scipy.spatial.distance import pdist

# TODO: Add tests that checks that the acq_max actually returns the maximum

@pytest.fixture
def target_func():
    return lambda x: sum(x)


@pytest.fixture
def random_state():
    return np.random.RandomState()


@pytest.fixture
def gp():
    return GaussianProcessRegressor()


@pytest.fixture
def target_space(target_func):
    return TargetSpace(target_func=target_func, pbounds={'x': (1, 4), 'y': (0, 3.)})


@pytest.fixture
def constrained_target_space(target_func):
    constraint_model = ConstraintModel(fun=lambda params: params['x'] + params['y'], lb=0.0, ub=1.0)
    return TargetSpace(target_func=target_func, pbounds={'x': (1, 4), 'y': (0, 3)}, constraint=constraint_model)


def test_base_acquisition():
    acq = acquisition.AcquisitionFunction()
    assert isinstance(acq.random_state, np.random.RandomState)
    acq = acquisition.AcquisitionFunction(random_state=42)
    assert isinstance(acq.random_state, np.random.RandomState)


def test_upper_confidence_bound(gp, target_space, random_state):
    acq = acquisition.UpperConfidenceBound(exploration_decay=0.5, exploration_decay_delay=2, kappa=1.0, random_state=random_state)
    assert acq.kappa == 1.0

    with pytest.raises(ValueError):
        acq.suggest(gp=gp, target_space=target_space)
    target_space.register(params={'x': 2.5, 'y': 0.5}, target=3.0)
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.kappa == 1.0
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.kappa == 0.5


def test_l_bfgs_fails(gp, target_space, random_state):
    acq = acquisition.AcquisitionFunction(random_state=random_state)

    def fun(x):
        try:
            return np.nan * np.zeros_like(x[:,0])
        except IndexError:
            return np.nan

    _, min_acq_l = acq._l_bfgs_b_minimize(fun, bounds=target_space.bounds, n_x_seeds=1)
    assert min_acq_l == np.inf
        
def test_upper_confidence_bound_with_constraints(gp, constrained_target_space, random_state):
    acq = acquisition.UpperConfidenceBound(random_state=random_state)

    constrained_target_space.register(params={'x': 2.5, 'y': 0.5}, target=3.0, constraint_value=0.5)
    with pytest.raises(acquisition.ConstraintNotSupportedError):
        acq.suggest(gp=gp, target_space=constrained_target_space)


def test_probability_of_improvement(gp, target_space, random_state):
    acq = acquisition.ProbabilityOfImprovement(exploration_decay=0.5, exploration_decay_delay=2, xi=0.01, random_state=random_state)
    assert acq.xi == 0.01

    with pytest.raises(ValueError):
        acq.suggest(gp=gp, target_space=target_space)

    target_space.register(params={'x': 2.5, 'y': 0.5}, target=3.0)
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.xi == 0.01
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.xi == 0.005


def test_expected_improvement(gp, target_space, random_state):
    acq = acquisition.ExpectedImprovement(exploration_decay=0.5, exploration_decay_delay=2, xi=0.01, random_state=random_state)
    assert acq.xi == 0.01

    with pytest.raises(ValueError):
        acq.suggest(gp=gp, target_space=target_space)
    target_space.register(params={'x': 2.5, 'y': 0.5}, target=3.0)
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.xi == 0.01
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.xi == 0.005


@pytest.mark.parametrize("strategy", [0., 'mean', 'min', 'max'])
def test_constant_liar(gp, target_space, target_func, random_state, strategy):
    base_acq = acquisition.UpperConfidenceBound(random_state=random_state)
    acq = acquisition.ConstantLiar(base_acquisition=base_acq, strategy=strategy, random_state=random_state)

    target_space.register(params={'x': 2.5, 'y': 0.5}, target=3.0)
    target_space.register(params={'x': 1.0, 'y': 1.5}, target=2.5)
    base_samples = np.array([base_acq.suggest(gp=gp, target_space=target_space) for _ in range(10)])
    samples = []

    assert len(acq.dummies) == 0
    for i in range(10):
        samples.append(acq.suggest(gp=gp, target_space=target_space))
        assert len(acq.dummies) == len(samples)

    samples = np.array(samples)
    print(samples)

    base_distance = pdist(base_samples, 'sqeuclidean').mean()
    distance = pdist(samples, 'sqeuclidean').mean()

    assert base_distance < distance

    for i in range(10):
        target_space.register(params={'x': samples[i][0], 'y': samples[i][1]}, target=target_func(samples[i]))

    acq.suggest(gp=gp, target_space=target_space)

    assert len(acq.dummies) == 1


def test_constant_liar_with_constraints(gp, constrained_target_space, random_state):
    base_acq = acquisition.UpperConfidenceBound(random_state=random_state)
    acq = acquisition.ConstantLiar(base_acquisition=base_acq, random_state=random_state)

    with pytest.raises(ValueError):
        acq.suggest(gp=gp, target_space=constrained_target_space)

    constrained_target_space.register(params={'x': 2.5, 'y': 0.5}, target=3.0, constraint_value=0.5)
    with pytest.raises(acquisition.ConstraintNotSupportedError):
        acq.suggest(gp=gp, target_space=constrained_target_space)


def test_kriging_believer(gp, target_space, target_func, random_state):
    base_acq = acquisition.UpperConfidenceBound(random_state=random_state)
    acq = acquisition.KrigingBeliever(base_acquisition=base_acq, random_state=random_state)

    target_space.register(params={'x': 2.5, 'y': 0.5}, target=3.0)
    target_space.register(params={'x': 1.0, 'y': 1.5}, target=2.5)
    base_samples = np.array([base_acq.suggest(gp=gp, target_space=target_space) for _ in range(10)])
    samples = []

    assert len(acq.dummies) == 0
    for i in range(10):
        samples.append(acq.suggest(gp=gp, target_space=target_space))
        assert len(acq.dummies) == len(samples)

    samples = np.array(samples)
    print(samples)

    base_distance = pdist(base_samples, 'sqeuclidean').mean()
    distance = pdist(samples, 'sqeuclidean').mean()

    assert base_distance < distance

    for i in range(10):
        target_space.register(params={'x': samples[i][0], 'y': samples[i][1]}, target=target_func(samples[i]))

    acq.suggest(gp=gp, target_space=target_space)

    assert len(acq.dummies) == 1


def test_kriging_believer_with_constraints(gp, constrained_target_space, target_func, random_state):
    base_acq = acquisition.ExpectedImprovement(xi=0.1,random_state=random_state)
    acq = acquisition.KrigingBeliever(base_acquisition=base_acq, random_state=random_state)

    constrained_target_space.register(params={'x': 2.5, 'y': 0.5}, target=3.0, constraint_value=0.5)
    constrained_target_space.register(params={'x': 1.0, 'y': 1.5}, target=2.5, constraint_value=0.5)
    base_samples = np.array([base_acq.suggest(gp=gp, target_space=constrained_target_space) for _ in range(10)])
    samples = []

    assert len(acq.dummies) == 0
    for i in range(10):
        samples.append(acq.suggest(gp=gp, target_space=constrained_target_space))
        assert len(acq.dummies) == len(samples)

    samples = np.array(samples)

    base_distance = pdist(base_samples, 'sqeuclidean').mean()
    distance = pdist(samples, 'sqeuclidean').mean()

    assert base_distance < distance

    for i in range(10):
        constrained_target_space.register(params={'x': samples[i][0], 'y': samples[i][1]}, target=target_func(samples[i]), constraint_value=0.5)

    acq.suggest(gp=gp, target_space=constrained_target_space)

    assert len(acq.dummies) == 1
