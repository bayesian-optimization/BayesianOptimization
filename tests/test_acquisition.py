from __future__ import annotations

import sys

import numpy as np
import pytest
from scipy.optimize import NonlinearConstraint
from scipy.spatial.distance import pdist
from sklearn.gaussian_process import GaussianProcessRegressor

from bayes_opt import BayesianOptimization, acquisition, exception
from bayes_opt.acquisition import (
    ConstantLiar,
    ExpectedImprovement,
    GPHedge,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from bayes_opt.constraint import ConstraintModel
from bayes_opt.target_space import TargetSpace


# Test fixtures
@pytest.fixture
def target_func_x_and_y():
    return lambda x, y: -((x - 1) ** 2) - (y - 2) ** 2


@pytest.fixture
def pbounds():
    return {"x": (-5, 5), "y": (-5, 5)}


@pytest.fixture
def constraint(constraint_func):
    return NonlinearConstraint(fun=constraint_func, lb=-1.0, ub=4.0)


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
    return TargetSpace(target_func=target_func, pbounds={"x": (1, 4), "y": (0, 3.0)})


@pytest.fixture
def constraint_func():
    return lambda x, y: x + y


@pytest.fixture
def constrained_target_space(target_func):
    constraint_model = ConstraintModel(fun=lambda params: params["x"] + params["y"], lb=0.0, ub=1.0)
    return TargetSpace(
        target_func=target_func, pbounds={"x": (1, 4), "y": (0, 3)}, constraint=constraint_model
    )


class MockAcquisition(acquisition.AcquisitionFunction):
    def __init__(self, random_state=None):
        super().__init__(random_state=random_state)

    def _get_acq(self, gp, constraint=None):
        def mock_acq(x: np.ndarray):
            return (3 - x[..., 0]) ** 2 + (1 - x[..., 1]) ** 2

        return mock_acq

    def base_acq(self, mean, std):
        pass

    def get_acquisition_params(self) -> dict:
        return {}

    def set_acquisition_params(self, params: dict) -> None:
        pass


def test_base_acquisition():
    acq = acquisition.UpperConfidenceBound()
    assert isinstance(acq.random_state, np.random.RandomState)
    acq = acquisition.UpperConfidenceBound(random_state=42)
    assert isinstance(acq.random_state, np.random.RandomState)


def test_acquisition_optimization(gp, target_space):
    acq = MockAcquisition(random_state=42)
    target_space.register(params={"x": 2.5, "y": 0.5}, target=3.0)
    res = acq.suggest(gp=gp, target_space=target_space)
    assert np.array([3.0, 1.0]) == pytest.approx(res)

    with pytest.raises(ValueError):
        acq.suggest(gp=gp, target_space=target_space, n_random=0, n_l_bfgs_b=0)


def test_acquisition_optimization_only_random(gp, target_space):
    acq = MockAcquisition(random_state=42)
    target_space.register(params={"x": 2.5, "y": 0.5}, target=3.0)
    res = acq.suggest(gp=gp, target_space=target_space, n_l_bfgs_b=0, n_random=10_000)
    # very lenient comparison as we're just considering random samples
    assert np.array([3.0, 1.0]) == pytest.approx(res, abs=1e-1, rel=1e-1)


def test_acquisition_optimization_only_l_bfgs_b(gp, target_space):
    acq = MockAcquisition(random_state=42)
    target_space.register(params={"x": 2.5, "y": 0.5}, target=3.0)
    res = acq.suggest(gp=gp, target_space=target_space, n_l_bfgs_b=10, n_random=0)
    assert np.array([3.0, 1.0]) == pytest.approx(res)


def test_upper_confidence_bound(gp, target_space, random_state):
    acq = acquisition.UpperConfidenceBound(
        exploration_decay=0.5, exploration_decay_delay=2, kappa=1.0, random_state=random_state
    )
    assert acq.kappa == 1.0

    # Test that the suggest method raises an error if the GP is unfitted
    with pytest.raises(
        exception.TargetSpaceEmptyError, match="Cannot suggest a point without previous samples"
    ):
        acq.suggest(gp=gp, target_space=target_space)

    target_space.register(params={"x": 2.5, "y": 0.5}, target=3.0)
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.kappa == 1.0
    acq.suggest(gp=gp, target_space=target_space)
    assert acq.kappa == 0.5


def test_smart_minimize_fails(target_space, random_state):
    acq = acquisition.UpperConfidenceBound(random_state=random_state)

    def fun(x):
        try:
            return np.nan * np.zeros_like(x[:, 0])
        except IndexError:
            return np.nan

    _, min_acq_l = acq._smart_minimize(fun, space=target_space, x_seeds=np.array([[2.5, 0.5]]))
    assert min_acq_l == np.inf


def test_upper_confidence_bound_with_constraints(gp, constrained_target_space, random_state):
    acq = acquisition.UpperConfidenceBound(random_state=random_state)

    constrained_target_space.register(params={"x": 2.5, "y": 0.5}, target=3.0, constraint_value=0.5)
    with pytest.raises(exception.ConstraintNotSupportedError):
        acq.suggest(gp=gp, target_space=constrained_target_space)


def test_probability_of_improvement(gp, target_space, random_state):
    acq = acquisition.ProbabilityOfImprovement(
        exploration_decay=0.5, exploration_decay_delay=2, xi=0.01, random_state=random_state
    )
    assert acq.xi == 0.01
    with pytest.raises(ValueError, match="y_max is not set"):
        acq.base_acq(0.0, 0.0)

    target_space.register(params={"x": 2.5, "y": 0.5}, target=3.0)
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
    acq = acquisition.ProbabilityOfImprovement(
        exploration_decay=0.5, exploration_decay_delay=2, xi=0.01, random_state=random_state
    )
    assert acq.xi == 0.01
    with pytest.raises(ValueError, match="y_max is not set"):
        acq.base_acq(0.0, 0.0)

    with pytest.raises(exception.TargetSpaceEmptyError):
        acq.suggest(gp=gp, target_space=constrained_target_space)

    constrained_target_space.register(params={"x": 2.5, "y": 0.5}, target=3.0, constraint_value=3.0)
    with pytest.raises(exception.NoValidPointRegisteredError):
        acq.suggest(gp=gp, target_space=constrained_target_space)

    constrained_target_space.register(params={"x": 1.0, "y": 0.0}, target=1.0, constraint_value=1.0)
    acq.suggest(gp=gp, target_space=constrained_target_space)


def test_expected_improvement(gp, target_space, random_state):
    acq = acquisition.ExpectedImprovement(
        exploration_decay=0.5, exploration_decay_delay=2, xi=0.01, random_state=random_state
    )
    assert acq.xi == 0.01

    with pytest.raises(ValueError, match="y_max is not set"):
        acq.base_acq(0.0, 0.0)

    target_space.register(params={"x": 2.5, "y": 0.5}, target=3.0)
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
    acq = acquisition.ExpectedImprovement(
        exploration_decay=0.5, exploration_decay_delay=2, xi=0.01, random_state=random_state
    )
    assert acq.xi == 0.01
    with pytest.raises(ValueError, match="y_max is not set"):
        acq.base_acq(0.0, 0.0)

    with pytest.raises(exception.TargetSpaceEmptyError):
        acq.suggest(gp=gp, target_space=constrained_target_space)

    constrained_target_space.register(params={"x": 2.5, "y": 0.5}, target=3.0, constraint_value=3.0)
    with pytest.raises(exception.NoValidPointRegisteredError):
        acq.suggest(gp=gp, target_space=constrained_target_space)

    constrained_target_space.register(params={"x": 1.0, "y": 0.0}, target=1.0, constraint_value=1.0)
    acq.suggest(gp=gp, target_space=constrained_target_space)


@pytest.mark.parametrize("strategy", [0.0, "mean", "min", "max"])
def test_constant_liar(gp, target_space, target_func, random_state, strategy):
    base_acq = acquisition.UpperConfidenceBound(random_state=random_state)
    acq = acquisition.ConstantLiar(base_acquisition=base_acq, strategy=strategy, random_state=random_state)

    target_space.register(params={"x": 2.5, "y": 0.5}, target=3.0)
    target_space.register(params={"x": 1.0, "y": 1.5}, target=2.5)
    base_samples = np.array([base_acq.suggest(gp=gp, target_space=target_space) for _ in range(10)])
    samples = []

    assert len(acq.dummies) == 0
    for _ in range(10):
        samples.append(acq.suggest(gp=gp, target_space=target_space))
        assert len(acq.dummies) == len(samples)

    samples = np.array(samples)
    print(samples)

    base_distance = pdist(base_samples, "sqeuclidean").mean()
    distance = pdist(samples, "sqeuclidean").mean()

    assert base_distance < distance

    for i in range(10):
        target_space.register(params={"x": samples[i][0], "y": samples[i][1]}, target=target_func(samples[i]))

    acq.suggest(gp=gp, target_space=target_space)

    assert len(acq.dummies) == 1


def test_constant_liar_invalid_strategy():
    with pytest.raises(ValueError):
        acquisition.ConstantLiar(acquisition.UpperConfidenceBound, strategy="definitely-an-invalid-strategy")


def test_constant_liar_with_constraints(gp, constrained_target_space, random_state):
    base_acq = acquisition.UpperConfidenceBound(random_state=random_state)
    acq = acquisition.ConstantLiar(base_acquisition=base_acq, random_state=random_state)

    with pytest.raises(exception.TargetSpaceEmptyError):
        acq.suggest(gp=gp, target_space=constrained_target_space)

    constrained_target_space.register(params={"x": 2.5, "y": 0.5}, target=3.0, constraint_value=0.5)
    with pytest.raises(exception.ConstraintNotSupportedError):
        acq.suggest(gp=gp, target_space=constrained_target_space)

    mean = random_state.rand(10)
    std = random_state.rand(10)
    assert (base_acq.base_acq(mean, std) == acq.base_acq(mean, std)).all()


def test_gp_hedge(random_state):
    acq = acquisition.GPHedge(
        base_acquisitions=[acquisition.UpperConfidenceBound(random_state=random_state)],
        random_state=random_state,
    )
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

    class MockGP1:
        def __init__(self, n):
            self.gains = np.zeros(n)

        def predict(self, x):
            rng = np.random.default_rng()
            res = rng.random(x.shape[0], np.float64)
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

    class MockGP2:
        def __init__(self, good_index=0):
            self.good_index = good_index

        def predict(self, x):
            print(x)
            res = -np.inf * np.ones_like(x)
            res[self.good_index] = 1.0
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
    with pytest.raises(exception.TargetSpaceEmptyError):
        acq.suggest(gp=gp, target_space=target_space)
    target_space.register(params={"x": 2.5, "y": 0.5}, target=3.0)

    for _ in range(5):
        p = acq.suggest(gp=gp, target_space=target_space)
        target_space.register(p, sum(p))


@pytest.mark.parametrize("kappa", [-1.0, -sys.float_info.epsilon, -np.inf])
def test_upper_confidence_bound_invalid_kappa_error(kappa: float):
    with pytest.raises(ValueError, match="kappa must be greater than or equal to 0."):
        acquisition.UpperConfidenceBound(kappa=kappa)


def verify_optimizers_match(optimizer1, optimizer2):
    """Helper function to verify two optimizers match."""
    assert len(optimizer1.space) == len(optimizer2.space)
    assert optimizer1.max["target"] == optimizer2.max["target"]
    assert optimizer1.max["params"] == optimizer2.max["params"]

    np.testing.assert_array_equal(optimizer1.space.params, optimizer2.space.params)
    np.testing.assert_array_equal(optimizer1.space.target, optimizer2.space.target)

    if optimizer1.is_constrained:
        np.testing.assert_array_equal(
            optimizer1.space._constraint_values, optimizer2.space._constraint_values
        )
        assert optimizer1.space._constraint.lb == optimizer2.space._constraint.lb
        assert optimizer1.space._constraint.ub == optimizer2.space._constraint.ub

    rng = np.random.default_rng()
    assert rng.bit_generator.state["state"]["state"] == rng.bit_generator.state["state"]["state"]

    kernel_params1 = optimizer1._gp.kernel.get_params()
    kernel_params2 = optimizer2._gp.kernel.get_params()
    for k in kernel_params1:
        assert (np.array(kernel_params1[k]) == np.array(kernel_params2[k])).all()

    suggestion1 = optimizer1.suggest()
    suggestion2 = optimizer2.suggest()
    assert suggestion1 == suggestion2, f"\nSuggestion 1: {suggestion1}\nSuggestion 2: {suggestion2}"


def test_integration_upper_confidence_bound(target_func_x_and_y, pbounds, tmp_path):
    """Test save/load integration with UpperConfidenceBound acquisition."""
    acquisition_function = UpperConfidenceBound(kappa=2.576)

    # Create and run first optimizer
    optimizer = BayesianOptimization(
        f=target_func_x_and_y,
        pbounds=pbounds,
        acquisition_function=acquisition_function,
        random_state=1,
        verbose=0,
    )
    optimizer.maximize(init_points=2, n_iter=3)

    # Save state
    state_path = tmp_path / "ucb_state.json"
    optimizer.save_state(state_path)

    # Create new optimizer and load state
    new_optimizer = BayesianOptimization(
        f=target_func_x_and_y,
        pbounds=pbounds,
        acquisition_function=UpperConfidenceBound(kappa=2.576),
        random_state=1,
        verbose=0,
    )
    new_optimizer.load_state(state_path)

    verify_optimizers_match(optimizer, new_optimizer)


def test_integration_probability_improvement(target_func_x_and_y, pbounds, tmp_path):
    """Test save/load integration with ProbabilityOfImprovement acquisition."""
    acquisition_function = ProbabilityOfImprovement(xi=0.01)

    optimizer = BayesianOptimization(
        f=target_func_x_and_y,
        pbounds=pbounds,
        acquisition_function=acquisition_function,
        random_state=1,
        verbose=0,
    )
    optimizer.maximize(init_points=2, n_iter=3)

    state_path = tmp_path / "pi_state.json"
    optimizer.save_state(state_path)

    new_optimizer = BayesianOptimization(
        f=target_func_x_and_y,
        pbounds=pbounds,
        acquisition_function=ProbabilityOfImprovement(xi=0.01),
        random_state=1,
        verbose=0,
    )
    new_optimizer.load_state(state_path)

    verify_optimizers_match(optimizer, new_optimizer)


def test_integration_expected_improvement(target_func_x_and_y, pbounds, tmp_path):
    """Test save/load integration with ExpectedImprovement acquisition."""
    acquisition_function = ExpectedImprovement(xi=0.01)

    optimizer = BayesianOptimization(
        f=target_func_x_and_y,
        pbounds=pbounds,
        acquisition_function=acquisition_function,
        random_state=1,
        verbose=0,
    )
    optimizer.maximize(init_points=2, n_iter=3)

    state_path = tmp_path / "ei_state.json"
    optimizer.save_state(state_path)

    new_optimizer = BayesianOptimization(
        f=target_func_x_and_y,
        pbounds=pbounds,
        acquisition_function=ExpectedImprovement(xi=0.01),
        random_state=1,
        verbose=0,
    )
    new_optimizer.load_state(state_path)

    verify_optimizers_match(optimizer, new_optimizer)


def test_integration_constant_liar(target_func_x_and_y, pbounds, tmp_path):
    """Test save/load integration with ConstantLiar acquisition."""
    base_acq = UpperConfidenceBound(kappa=2.576)
    acquisition_function = ConstantLiar(base_acquisition=base_acq)

    optimizer = BayesianOptimization(
        f=target_func_x_and_y,
        pbounds=pbounds,
        acquisition_function=acquisition_function,
        random_state=1,
        verbose=0,
    )
    optimizer.maximize(init_points=2, n_iter=3)

    state_path = tmp_path / "cl_state.json"
    optimizer.save_state(state_path)

    new_optimizer = BayesianOptimization(
        f=target_func_x_and_y,
        pbounds=pbounds,
        acquisition_function=ConstantLiar(base_acquisition=UpperConfidenceBound(kappa=2.576)),
        random_state=1,
        verbose=0,
    )
    new_optimizer.load_state(state_path)

    verify_optimizers_match(optimizer, new_optimizer)


def test_integration_gp_hedge(target_func_x_and_y, pbounds, tmp_path):
    """Test save/load integration with GPHedge acquisition."""
    base_acquisitions = [
        UpperConfidenceBound(kappa=2.576),
        ProbabilityOfImprovement(xi=0.01),
        ExpectedImprovement(xi=0.01),
    ]
    acquisition_function = GPHedge(base_acquisitions=base_acquisitions)

    optimizer = BayesianOptimization(
        f=target_func_x_and_y,
        pbounds=pbounds,
        acquisition_function=acquisition_function,
        random_state=1,
        verbose=0,
    )
    optimizer.maximize(init_points=2, n_iter=3)

    state_path = tmp_path / "gphedge_state.json"
    optimizer.save_state(state_path)

    new_base_acquisitions = [
        UpperConfidenceBound(kappa=2.576),
        ProbabilityOfImprovement(xi=0.01),
        ExpectedImprovement(xi=0.01),
    ]
    new_optimizer = BayesianOptimization(
        f=target_func_x_and_y,
        pbounds=pbounds,
        acquisition_function=GPHedge(base_acquisitions=new_base_acquisitions),
        random_state=1,
        verbose=0,
    )
    new_optimizer.load_state(state_path)

    verify_optimizers_match(optimizer, new_optimizer)


def test_integration_constrained(target_func_x_and_y, pbounds, constraint, tmp_path):
    """Test save/load integration with constraints."""
    acquisition_function = ExpectedImprovement(xi=0.01)

    optimizer = BayesianOptimization(
        f=target_func_x_and_y,
        pbounds=pbounds,
        acquisition_function=acquisition_function,
        constraint=constraint,
        random_state=1,
        verbose=0,
    )
    optimizer.maximize(init_points=2, n_iter=3)

    state_path = tmp_path / "constrained_state.json"
    optimizer.save_state(state_path)

    new_optimizer = BayesianOptimization(
        f=target_func_x_and_y,
        pbounds=pbounds,
        acquisition_function=ExpectedImprovement(xi=0.01),
        constraint=constraint,
        random_state=1,
        verbose=0,
    )
    new_optimizer.load_state(state_path)

    verify_optimizers_match(optimizer, new_optimizer)


def test_custom_acquisition_without_get_params():
    """Test that a custom acquisition function without get_acquisition_params raises NotImplementedError."""

    class CustomAcqWithoutGetParams(acquisition.AcquisitionFunction):
        def __init__(self, random_state=None):
            super().__init__(random_state=random_state)

        def base_acq(self, mean, std):
            return mean + std

        def set_acquisition_params(self, params):
            pass

    acq = CustomAcqWithoutGetParams()
    with pytest.raises(
        NotImplementedError,
        match="Custom AcquisitionFunction subclasses must implement their own get_acquisition_params method",
    ):
        acq.get_acquisition_params()


def test_custom_acquisition_without_set_params():
    """Test that a custom acquisition function without set_acquisition_params raises NotImplementedError."""

    class CustomAcqWithoutSetParams(acquisition.AcquisitionFunction):
        def __init__(self, random_state=None):
            super().__init__(random_state=random_state)

        def base_acq(self, mean, std):
            return mean + std

        def get_acquisition_params(self):
            return {}

    acq = CustomAcqWithoutSetParams()
    with pytest.raises(
        NotImplementedError,
        match="Custom AcquisitionFunction subclasses must implement their own set_acquisition_params method",
    ):
        acq.set_acquisition_params(params={})
