import pytest
import numpy as np
from bayes_opt import acquisition
from sklearn.gaussian_process import GaussianProcessRegressor
from bayes_opt.target_space import TargetSpace
from bayes_opt.constraint import ConstraintModel
from scipy.spatial.distance import pdist
from bayes_opt import acquisition 
# TODO: Add tests that checks that the acq_max actually returns the maximum

@pytest.fixture
def target_func():
    return lambda x: sum(x)


@pytest.fixture
def random_state():
    return np.random.RandomState()


@pytest.fixture
def gp(random_state):
    return GaussianProcessRegressor(random_state=random_state)


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

    # Test that the suggest method raises an error if the GP is unfitted
    with pytest.raises(acquisition.TargetSpaceEmptyError, match="Cannot suggest a point without previous samples"):
        acq.suggest(gp=gp, target_space=target_space)

    target_space.register(params={'x': 2.5, 'y': 0.5}, target=3.0)
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.kappa == 1.0
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.kappa == 0.5


def test_l_bfgs_fails(target_space, random_state):
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
    with pytest.raises(ValueError, match="y_max is not set"):
        acq.base_acq(0.0, 0.0)

    target_space.register(params={'x': 2.5, 'y': 0.5}, target=3.0)
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.xi == 0.01
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.xi == 0.005

    # no decay
    acq = acquisition.ProbabilityOfImprovement(exploration_decay=None, xi=0.01, random_state=random_state)
    assert acq.xi == 0.01
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.xi == 0.01
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.xi == 0.01

def test_probability_of_improvement_with_constraints(gp, constrained_target_space, random_state):
    acq = acquisition.ProbabilityOfImprovement(exploration_decay=0.5, exploration_decay_delay=2, xi=0.01, random_state=random_state)
    assert acq.xi == 0.01
    with pytest.raises(ValueError, match="y_max is not set"):
        acq.base_acq(0.0, 0.0)

    with pytest.raises(acquisition.TargetSpaceEmptyError):
        acq.suggest(gp=gp, target_space=constrained_target_space)

    constrained_target_space.register(params={'x': 2.5, 'y': 0.5}, target=3.0, constraint_value=3.0)
    with pytest.raises(acquisition.NoValidPointRegisteredError):
        acq.suggest(gp=gp, target_space=constrained_target_space)

    constrained_target_space.register(params={'x': 1.0, 'y': 0.0}, target=1.0, constraint_value=1.0)
    acq.suggest(gp=gp, target_space=constrained_target_space)


def test_expected_improvement(gp, target_space, random_state):
    acq = acquisition.ExpectedImprovement(exploration_decay=0.5, exploration_decay_delay=2, xi=0.01, random_state=random_state)
    assert acq.xi == 0.01

    with pytest.raises(ValueError, match="y_max is not set"):
        acq.base_acq(0.0, 0.0)

    target_space.register(params={'x': 2.5, 'y': 0.5}, target=3.0)
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.xi == 0.01
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.xi == 0.005

    acq = acquisition.ExpectedImprovement(exploration_decay=None, xi=0.01, random_state=random_state)
    assert acq.xi == 0.01
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.xi == 0.01
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.xi == 0.01


def test_expected_improvement_with_constraints(gp, constrained_target_space, random_state):
    acq = acquisition.ExpectedImprovement(exploration_decay=0.5, exploration_decay_delay=2, xi=0.01, random_state=random_state)
    assert acq.xi == 0.01
    with pytest.raises(ValueError, match="y_max is not set"):
        acq.base_acq(0.0, 0.0)

    with pytest.raises(acquisition.TargetSpaceEmptyError):
        acq.suggest(gp=gp, target_space=constrained_target_space)

    constrained_target_space.register(params={'x': 2.5, 'y': 0.5}, target=3.0, constraint_value=3.0)
    with pytest.raises(acquisition.NoValidPointRegisteredError):
        acq.suggest(gp=gp, target_space=constrained_target_space)

    constrained_target_space.register(params={'x': 1.0, 'y': 0.0}, target=1.0, constraint_value=1.0)
    acq.suggest(gp=gp, target_space=constrained_target_space)


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

    with pytest.raises(acquisition.TargetSpaceEmptyError):
        acq.suggest(gp=gp, target_space=constrained_target_space)

    constrained_target_space.register(params={'x': 2.5, 'y': 0.5}, target=3.0, constraint_value=0.5)
    with pytest.raises(acquisition.ConstraintNotSupportedError):
        acq.suggest(gp=gp, target_space=constrained_target_space)

    mean = random_state.rand(10)
    std = random_state.rand(10)
    assert (base_acq.base_acq(mean, std) == acq.base_acq(mean, std)).all()


def test_gp_hedge(random_state):
    acq = acquisition.GPHedge(base_acquisitions=[acquisition.UpperConfidenceBound(random_state=random_state)], random_state=random_state)
    with pytest.raises(TypeError, match="GPHedge base acquisition function is ambiguous"):
        acq.base_acq(0.0, 0.0)

    base_acq1 = acquisition.UpperConfidenceBound()
    base_acq2 = acquisition.ProbabilityOfImprovement(xi=0.01)
    base_acquisitions = [base_acq1, base_acq2]
    acq = acquisition.GPHedge(base_acquisitions=base_acquisitions)

    mean = random_state.rand(10)
    std = random_state.rand(10)

    base_acq2.y_max = 1.0
    assert (acq.base_acquisitions[0].base_acq(mean, std) == base_acq1.base_acq(mean, std)).all()
    assert (acq.base_acquisitions[1].base_acq(mean, std) == base_acq2.base_acq(mean, std)).all()


def test_gphedge_update_gains(random_state):
    base_acq1 = acquisition.UpperConfidenceBound(random_state=random_state)
    base_acq2 = acquisition.ProbabilityOfImprovement(xi=0.01, random_state=random_state)
    base_acquisitions = [base_acq1, base_acq2]

    acq = acquisition.GPHedge(base_acquisitions=base_acquisitions, random_state=random_state)

    class MockGP1():
        def __init__(self, n):
            self.gains = np.zeros(n)

        def predict(self, x):
            res = np.random.rand(x.shape[0])
            self.gains += res
            return res

    mock_gp = MockGP1(len(base_acquisitions))
    for _ in range(10):
        acq.previous_candidates = np.zeros(len(base_acquisitions))
        acq._update_gains(mock_gp)
        assert (mock_gp.gains == acq.gains).all()


def test_gphedge_softmax_sampling(random_state):
    base_acq1 = acquisition.UpperConfidenceBound(random_state=random_state)
    base_acq2 = acquisition.ProbabilityOfImprovement(xi=0.01, random_state=random_state)
    base_acquisitions = [base_acq1, base_acq2]

    acq = acquisition.GPHedge(base_acquisitions=base_acquisitions, random_state=random_state)

    class MockGP2():
        def __init__(self, good_index=0):
            self.good_index = good_index

        def predict(self, x):
            print(x)
            res = -np.inf * np.ones_like(x)
            res[self.good_index] = 1.
            return res

    for good_index in [0, 1]:
        acq = acquisition.GPHedge(base_acquisitions=base_acquisitions)
        acq.previous_candidates = np.zeros(len(base_acquisitions))
        acq._update_gains(MockGP2(good_index=good_index))
        assert good_index == acq._sample_idx_from_softmax_gains()


def test_gphedge_integration(gp, target_space, random_state):
    base_acq1 = acquisition.UpperConfidenceBound(random_state=random_state)
    base_acq2 = acquisition.ProbabilityOfImprovement(xi=0.01, random_state=random_state)
    base_acquisitions = [base_acq1, base_acq2]

    acq = acquisition.GPHedge(base_acquisitions=base_acquisitions, random_state=random_state)
    assert acq.base_acquisitions == base_acquisitions
    with pytest.raises(acquisition.TargetSpaceEmptyError):
        acq.suggest(gp=gp, target_space=target_space)
    target_space.register(params={'x': 2.5, 'y': 0.5}, target=3.0)

    for _ in range(5):
        p = acq.suggest(gp=gp, target_space=target_space)
        target_space.register(p, sum(p))
