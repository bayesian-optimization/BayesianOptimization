from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import NonlinearConstraint

from bayes_opt import BayesianOptimization, acquisition
from bayes_opt.acquisition import AcquisitionFunction
from bayes_opt.domain_reduction import SequentialDomainReductionTransformer
from bayes_opt.exception import NotUniqueError
from bayes_opt.parameter import BayesParameter
from bayes_opt.target_space import TargetSpace
from bayes_opt.util import ensure_rng


class FixedPerimeterTriangleParameter(BayesParameter):
    def __init__(self, name: str, bounds, perimeter) -> None:
        super().__init__(name, bounds)
        self.perimeter = perimeter

    @property
    def is_continuous(self):
        return True

    def random_sample(self, n_samples: int, random_state):
        random_state = ensure_rng(random_state)
        samples = []
        while len(samples) < n_samples:
            samples_ = random_state.dirichlet(np.ones(3), n_samples)
            samples_ = samples_ * self.perimeter  # scale samples by perimeter

            samples_ = samples_[
                np.all((self.bounds[:, 0] <= samples_) & (samples_ <= self.bounds[:, 1]), axis=-1)
            ]
            samples.extend(np.atleast_2d(samples_))
        return np.array(samples[:n_samples])

    def to_float(self, value):
        return value

    def to_param(self, value):
        return value * self.perimeter / sum(value)

    def kernel_transform(self, value):
        return value * self.perimeter / np.sum(value, axis=-1, keepdims=True)

    def to_string(self, value, str_len: int) -> str:
        len_each = (str_len - 2) // 3
        str_ = "|".join([f"{float(np.round(value[i], 4))}"[:len_each] for i in range(3)])
        return str_.ljust(str_len)

    @property
    def dim(self):
        return 3  # as we have three float values, each representing the length of one side.


def area_of_triangle(sides):
    a, b, c = sides
    s = np.sum(sides, axis=-1)  # perimeter
    return np.sqrt(s * (s - a) * (s - b) * (s - c))


def target_func(**kwargs):
    # arbitrary target func
    return sum(kwargs.values())


PBOUNDS = {"p1": (0, 10), "p2": (0, 10)}


def test_properties():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert isinstance(optimizer.space, TargetSpace)
    assert isinstance(optimizer.acquisition_function, AcquisitionFunction)
    # constraint present tested in test_constraint.py
    assert optimizer.constraint is None


def test_register():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer.space) == 0

    optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    assert len(optimizer.res) == 1
    assert len(optimizer.space) == 1

    optimizer.space.register(params=np.array([5, 4]), target=9)
    assert len(optimizer.res) == 2
    assert len(optimizer.space) == 2

    with pytest.raises(NotUniqueError):
        optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    with pytest.raises(NotUniqueError):
        optimizer.register(params={"p1": 5, "p2": 4}, target=9)


def test_probe_lazy():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    optimizer.probe(params={"p1": 1, "p2": 2}, lazy=True)
    assert len(optimizer.space) == 0
    assert len(optimizer._queue) == 1

    optimizer.probe(params={"p1": 6, "p2": 2}, lazy=True)
    assert len(optimizer.space) == 0
    assert len(optimizer._queue) == 2

    optimizer.probe(params={"p1": 6, "p2": 2}, lazy=True)
    assert len(optimizer.space) == 0
    assert len(optimizer._queue) == 3


def test_probe_eager():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1, allow_duplicate_points=True)

    optimizer.probe(params={"p1": 1, "p2": 2}, lazy=False)
    assert len(optimizer.space) == 1
    assert len(optimizer._queue) == 0
    assert optimizer.max["target"] == 3
    assert optimizer.max["params"] == {"p1": 1, "p2": 2}

    optimizer.probe(params={"p1": 3, "p2": 3}, lazy=False)
    assert len(optimizer.space) == 2
    assert len(optimizer._queue) == 0
    assert optimizer.max["target"] == 6
    assert optimizer.max["params"] == {"p1": 3, "p2": 3}

    optimizer.probe(params={"p1": 3, "p2": 3}, lazy=False)
    assert len(optimizer.space) == 3
    assert len(optimizer._queue) == 0
    assert optimizer.max["target"] == 6
    assert optimizer.max["params"] == {"p1": 3, "p2": 3}


def test_suggest_at_random():
    acq = acquisition.ProbabilityOfImprovement(xi=0)
    optimizer = BayesianOptimization(target_func, PBOUNDS, acq, random_state=1)

    for _ in range(50):
        sample = optimizer.space.params_to_array(optimizer.suggest())
        assert len(sample) == optimizer.space.dim
        assert all(sample >= optimizer.space.bounds[:, 0])
        assert all(sample <= optimizer.space.bounds[:, 1])


def test_suggest_with_one_observation():
    acq = acquisition.UpperConfidenceBound(kappa=5)
    optimizer = BayesianOptimization(target_func, PBOUNDS, acq, random_state=1)

    optimizer.register(params={"p1": 1, "p2": 2}, target=3)

    for _ in range(5):
        sample = optimizer.space.params_to_array(optimizer.suggest())
        assert len(sample) == optimizer.space.dim
        assert all(sample >= optimizer.space.bounds[:, 0])
        assert all(sample <= optimizer.space.bounds[:, 1])

    # suggestion = optimizer.suggest(util)
    # for _ in range(5):
    #     new_suggestion = optimizer.suggest(util)
    #     assert suggestion == new_suggestion


def test_prime_queue_all_empty():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer._queue) == 0
    assert len(optimizer.space) == 0

    optimizer._prime_queue(init_points=0)
    assert len(optimizer._queue) == 1
    assert len(optimizer.space) == 0


def test_prime_queue_empty_with_init():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer._queue) == 0
    assert len(optimizer.space) == 0

    optimizer._prime_queue(init_points=5)
    assert len(optimizer._queue) == 5
    assert len(optimizer.space) == 0


def test_prime_queue_with_register():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer._queue) == 0
    assert len(optimizer.space) == 0

    optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    optimizer._prime_queue(init_points=0)
    assert len(optimizer._queue) == 0
    assert len(optimizer.space) == 1


def test_prime_queue_with_register_and_init():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer._queue) == 0
    assert len(optimizer.space) == 0

    optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    optimizer._prime_queue(init_points=3)
    assert len(optimizer._queue) == 3
    assert len(optimizer.space) == 1


def test_set_bounds():
    pbounds = {"p1": (0, 1), "p3": (0, 3), "p2": (0, 2), "p4": (0, 4)}
    optimizer = BayesianOptimization(target_func, pbounds, random_state=1)

    # Ignore unknown keys
    optimizer.set_bounds({"other": (7, 8)})
    assert all(optimizer.space.bounds[:, 0] == np.array([0, 0, 0, 0]))
    assert all(optimizer.space.bounds[:, 1] == np.array([1, 3, 2, 4]))

    # Update bounds accordingly
    optimizer.set_bounds({"p2": (1, 8)})
    assert all(optimizer.space.bounds[:, 0] == np.array([0, 0, 1, 0]))
    assert all(optimizer.space.bounds[:, 1] == np.array([1, 3, 8, 4]))


def test_set_gp_params():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert optimizer._gp.alpha == 1e-6
    assert optimizer._gp.n_restarts_optimizer == 5

    optimizer.set_gp_params(alpha=1e-2)
    assert optimizer._gp.alpha == 1e-2
    assert optimizer._gp.n_restarts_optimizer == 5

    optimizer.set_gp_params(n_restarts_optimizer=7)
    assert optimizer._gp.alpha == 1e-2
    assert optimizer._gp.n_restarts_optimizer == 7


def test_maximize():
    acq = acquisition.UpperConfidenceBound()
    optimizer = BayesianOptimization(
        target_func, PBOUNDS, acq, random_state=np.random.RandomState(1), allow_duplicate_points=True
    )

    # Test initial maximize with no init_points and n_iter
    optimizer.maximize(init_points=0, n_iter=0)
    assert not optimizer._queue
    assert len(optimizer.space) == 1  # Even with no init_points, we should have at least one point

    # Test after setting GP parameters
    optimizer.set_gp_params(alpha=1e-2)
    optimizer.maximize(init_points=2, n_iter=0)
    assert not optimizer._queue
    assert len(optimizer.space) == 3  # Previously had 1, add 2 more from init_points
    assert optimizer._gp.alpha == 1e-2

    # Test with additional iterations
    optimizer.maximize(init_points=0, n_iter=2)
    assert not optimizer._queue
    assert len(optimizer.space) == 5  # Previously had 3, add 2 more from n_iter


def test_define_wrong_transformer():
    with pytest.raises(TypeError):
        BayesianOptimization(
            target_func, PBOUNDS, random_state=np.random.RandomState(1), bounds_transformer=3
        )


def test_single_value_objective():
    """
    As documented [here](https://github.com/scipy/scipy/issues/16898)
    scipy is changing the way they handle 1D objectives inside minimize.
    This is a simple test to make sure our tests fail if scipy updates this
    in future
    """
    pbounds = {"x": (-10, 10)}

    optimizer = BayesianOptimization(f=lambda x: x * 3, pbounds=pbounds, verbose=2, random_state=1)
    optimizer.maximize(init_points=2, n_iter=3)


def test_pickle():
    """
    several users have asked that the BO object be 'pickalable'
    This tests that this is the case
    """
    optimizer = BayesianOptimization(f=None, pbounds={"x": (-10, 10)}, verbose=2, random_state=1)
    test_dump = Path("test_dump.obj")
    with test_dump.open("wb") as filehandler:
        pickle.dump(optimizer, filehandler)
    test_dump.unlink()


def test_duplicate_points():
    """
    The default behavior of this code is to not enable duplicate points in the target space,
    however there are situations in which you may want this, particularly optimization in high
    noise situations. In that case one can set allow_duplicate_points to be True.
    This tests the behavior of the code around duplicate points under several scenarios
    """
    # test manual registration of duplicate points (should generate error)
    acq = acquisition.UpperConfidenceBound(kappa=5.0)  # kappa determines explore/Exploitation ratio
    optimizer = BayesianOptimization(f=None, pbounds={"x": (-2, 2)}, acquisition_function=acq, random_state=1)
    next_point_to_probe = optimizer.suggest()
    target = 1
    # register once (should work)
    optimizer.register(params=next_point_to_probe, target=target)
    # register twice (should throw error)
    try:
        optimizer.register(params=next_point_to_probe, target=target)
        duplicate_point_error = None  # should be overwritten below
    except Exception as e:
        duplicate_point_error = e

    assert isinstance(duplicate_point_error, NotUniqueError)

    # OK, now let's test that it DOESNT fail when allow_duplicate_points=True
    optimizer = BayesianOptimization(
        f=None, pbounds={"x": (-2, 2)}, random_state=1, allow_duplicate_points=True
    )
    optimizer.register(params=next_point_to_probe, target=target)
    # and again (should throw warning)
    optimizer.register(params=next_point_to_probe, target=target)


def test_save_load_state(tmp_path):
    """Test saving and loading optimizer state."""
    # Initialize and run original optimizer
    optimizer = BayesianOptimization(f=target_func, pbounds=PBOUNDS, random_state=1, verbose=0)
    optimizer.maximize(init_points=2, n_iter=3)

    # Save state
    state_path = tmp_path / "optimizer_state.json"
    optimizer.save_state(state_path)

    # Create new optimizer and load state
    new_optimizer = BayesianOptimization(f=target_func, pbounds=PBOUNDS, random_state=1, verbose=0)
    new_optimizer.load_state(state_path)

    # Test that key properties match
    assert len(optimizer.space) == len(new_optimizer.space)
    assert optimizer.max["target"] == new_optimizer.max["target"]
    assert optimizer.max["params"] == new_optimizer.max["params"]
    np.testing.assert_array_equal(optimizer.space.params, new_optimizer.space.params)
    np.testing.assert_array_equal(optimizer.space.target, new_optimizer.space.target)


def test_save_load_w_categorical_params(tmp_path):
    """Test saving and loading optimizer state with categorical parameters."""

    def str_target_func(param1: str, param2: str) -> float:
        # Simple function that maps strings to numbers
        value_map = {"low": 1.0, "medium": 2.0, "high": 3.0}
        return value_map[param1] + value_map[param2]

    str_pbounds = {"param1": ["low", "medium", "high"], "param2": ["low", "medium", "high"]}

    optimizer = BayesianOptimization(f=str_target_func, pbounds=str_pbounds, random_state=1, verbose=0)

    optimizer.maximize(init_points=2, n_iter=3)

    state_path = tmp_path / "optimizer_state.json"
    optimizer.save_state(state_path)

    new_optimizer = BayesianOptimization(f=str_target_func, pbounds=str_pbounds, random_state=1, verbose=0)
    new_optimizer.load_state(state_path)

    assert len(optimizer.space) == len(new_optimizer.space)
    assert optimizer.max["target"] == new_optimizer.max["target"]
    assert optimizer.max["params"] == new_optimizer.max["params"]
    for i in range(len(optimizer.space)):
        assert isinstance(optimizer.res[i]["params"]["param1"], str)
        assert isinstance(optimizer.res[i]["params"]["param2"], str)
        assert isinstance(new_optimizer.res[i]["params"]["param1"], str)
        assert isinstance(new_optimizer.res[i]["params"]["param2"], str)
        assert optimizer.res[i]["params"] == new_optimizer.res[i]["params"]


def test_suggest_point_returns_same_point(tmp_path):
    """Check that suggest returns same point after save/load."""
    optimizer = BayesianOptimization(f=target_func, pbounds=PBOUNDS, random_state=1, verbose=0)
    optimizer.maximize(init_points=2, n_iter=3)

    state_path = tmp_path / "optimizer_state.json"
    optimizer.save_state(state_path)

    new_optimizer = BayesianOptimization(f=target_func, pbounds=PBOUNDS, random_state=1, verbose=0)
    new_optimizer.load_state(state_path)

    # Both optimizers should suggest the same point
    suggestion1 = optimizer.suggest()
    suggestion2 = new_optimizer.suggest()
    assert suggestion1 == suggestion2


def test_save_load_random_state(tmp_path):
    """Test that random state is properly preserved."""
    # Initialize optimizer
    optimizer = BayesianOptimization(f=target_func, pbounds=PBOUNDS, random_state=1, verbose=0)

    # Register a point before saving
    optimizer.probe(params={"p1": 1, "p2": 2}, lazy=False)

    # Save state
    state_path = tmp_path / "optimizer_state.json"
    optimizer.save_state(state_path)

    # Create new optimizer with same configuration
    new_optimizer = BayesianOptimization(f=target_func, pbounds=PBOUNDS, random_state=1, verbose=0)
    new_optimizer.load_state(state_path)

    # Both optimizers should suggest the same point
    suggestion1 = optimizer.suggest()
    suggestion2 = new_optimizer.suggest()
    assert suggestion1 == suggestion2


def test_save_load_unused_optimizer(tmp_path):
    """Test saving and loading optimizer state with unused optimizer."""
    optimizer = BayesianOptimization(f=target_func, pbounds=PBOUNDS, random_state=1, verbose=0)

    # Test that saving without samples does not raise an error
    optimizer.save_state(tmp_path / "unprobed_optimizer_state.json")

    # Check that we load the original state
    first_suggestion = optimizer.suggest()
    optimizer.load_state(tmp_path / "unprobed_optimizer_state.json")
    assert optimizer.suggest() == first_suggestion

    # Save an optimizer state with a probed point
    optimizer.probe(params={"p1": 1, "p2": 2}, lazy=False)
    optimizer.save_state(tmp_path / "optimizer_state.json")

    new_optimizer = BayesianOptimization(f=target_func, pbounds=PBOUNDS, random_state=1, verbose=0)
    new_optimizer.load_state(tmp_path / "optimizer_state.json")

    assert len(optimizer.space) == len(new_optimizer.space)
    assert optimizer.max["target"] == new_optimizer.max["target"]
    assert optimizer.max["params"] == new_optimizer.max["params"]
    np.testing.assert_array_equal(optimizer.space.params, new_optimizer.space.params)
    np.testing.assert_array_equal(optimizer.space.target, new_optimizer.space.target)

    """Test saving and loading optimizer state with constraints."""

    def constraint_func(x: float, y: float) -> float:
        return x + y  # Simple constraint: sum of parameters should be within bounds

    constraint = NonlinearConstraint(fun=constraint_func, lb=0.0, ub=3.0)

    # Initialize optimizer with constraint
    optimizer = BayesianOptimization(
        f=target_func, pbounds={"x": (-1, 3), "y": (0, 5)}, constraint=constraint, random_state=1, verbose=0
    )

    # Register some points, some that satisfy constraint and some that don't
    optimizer.register(
        params={"x": 1.0, "y": 1.0},  # Satisfies constraint: sum = 2.0
        target=2.0,
        constraint_value=2.0,
    )
    optimizer.register(
        params={"x": 2.0, "y": 2.0},  # Violates constraint: sum = 4.0
        target=4.0,
        constraint_value=4.0,
    )
    optimizer.register(
        params={"x": 0.5, "y": 0.5},  # Satisfies constraint: sum = 1.0
        target=1.0,
        constraint_value=1.0,
    )

    state_path = tmp_path / "optimizer_state.json"
    optimizer.save_state(state_path)

    new_optimizer = BayesianOptimization(
        f=target_func, pbounds={"x": (-1, 3), "y": (0, 5)}, constraint=constraint, random_state=1, verbose=0
    )
    new_optimizer.load_state(state_path)

    # Test that key properties match
    assert len(optimizer.space) == len(new_optimizer.space)
    assert optimizer.max["target"] == new_optimizer.max["target"]
    assert optimizer.max["params"] == new_optimizer.max["params"]
    np.testing.assert_array_equal(optimizer.space.params, new_optimizer.space.params)
    np.testing.assert_array_equal(optimizer.space.target, new_optimizer.space.target)

    # Test that constraint values were properly saved and loaded
    np.testing.assert_array_equal(optimizer.space._constraint_values, new_optimizer.space._constraint_values)

    # Test that both optimizers suggest the same point (should respect constraints)
    suggestion1 = optimizer.suggest()
    suggestion2 = new_optimizer.suggest()
    assert suggestion1 == suggestion2

    # Verify that suggested point satisfies constraint
    constraint_value = constraint_func(**suggestion1)
    assert 0.0 <= constraint_value <= 3.0, "Suggested point violates constraint"


def test_save_load_w_domain_reduction(tmp_path):
    """Test saving and loading optimizer state with domain reduction transformer."""
    # Initialize optimizer with bounds transformer
    bounds_transformer = SequentialDomainReductionTransformer()
    optimizer = BayesianOptimization(
        f=target_func, pbounds=PBOUNDS, random_state=1, verbose=0, bounds_transformer=bounds_transformer
    )

    # Run some iterations to trigger domain reduction
    optimizer.maximize(init_points=2, n_iter=3)

    # Save state
    state_path = tmp_path / "optimizer_state.json"
    optimizer.save_state(state_path)

    # Create new optimizer with same configuration
    new_bounds_transformer = SequentialDomainReductionTransformer()
    new_optimizer = BayesianOptimization(
        f=target_func, pbounds=PBOUNDS, random_state=1, verbose=0, bounds_transformer=new_bounds_transformer
    )
    new_optimizer.load_state(state_path)

    # Both optimizers should probe the same point
    point = {"p1": 1.5, "p2": 0.5}
    probe1 = optimizer.probe(point)
    probe2 = new_optimizer.probe(point)
    assert probe1 == probe2

    # Both optimizers should suggest the same point
    suggestion1 = optimizer.suggest()
    suggestion2 = new_optimizer.suggest()
    assert suggestion1 == suggestion2

    # Verify that the transformed bounds match
    assert optimizer._space.bounds.tolist() == new_optimizer._space.bounds.tolist()


def test_save_load_w_custom_parameter(tmp_path):
    """Test saving and loading optimizer state with custom parameter types."""

    # Create parameter and bounds
    param = FixedPerimeterTriangleParameter(
        name="sides", bounds=np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]), perimeter=1.0
    )
    pbounds = {"sides": param}

    # Print initial pbounds
    print("\nOriginal pbounds:")
    print(pbounds)

    # Initialize first optimizer
    optimizer = BayesianOptimization(f=area_of_triangle, pbounds=pbounds, random_state=1, verbose=0)

    # Run iterations and immediately save state
    optimizer.maximize(init_points=2, n_iter=5)

    # Force GP update before saving
    optimizer._gp.fit(optimizer.space.params, optimizer.space.target)

    # Save state
    state_path = tmp_path / "optimizer_state.json"
    optimizer.save_state(state_path)

    # Create new optimizer and load state
    new_optimizer = BayesianOptimization(f=area_of_triangle, pbounds=pbounds, random_state=1, verbose=0)
    new_optimizer.load_state(state_path)

    # Test that key properties match
    assert len(optimizer.space) == len(new_optimizer.space)
    assert optimizer.max["target"] == new_optimizer.max["target"]
    np.testing.assert_array_almost_equal(
        optimizer.max["params"]["sides"], new_optimizer.max["params"]["sides"], decimal=10
    )

    # Test that all historical data matches
    for i in range(len(optimizer.space)):
        np.testing.assert_array_almost_equal(
            optimizer.space.params[i], new_optimizer.space.params[i], decimal=10
        )
        assert optimizer.space.target[i] == new_optimizer.space.target[i]
        np.testing.assert_array_almost_equal(
            optimizer.res[i]["params"]["sides"], new_optimizer.res[i]["params"]["sides"], decimal=10
        )
        assert optimizer.res[i]["target"] == new_optimizer.res[i]["target"]

    # Test that multiple subsequent suggestions match
    for _ in range(5):
        suggestion1 = optimizer.suggest()
        suggestion2 = new_optimizer.suggest()
        np.testing.assert_array_almost_equal(suggestion1["sides"], suggestion2["sides"], decimal=7)


def test_predict():
    """Test the predict method of the optimizer."""
    optimizer = BayesianOptimization(f=target_func, pbounds=PBOUNDS, random_state=1, verbose=0)

    # Register some points
    optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    optimizer.register(params={"p1": 4, "p2": 5}, target=9)
    optimizer.register(params={"p1": 7, "p2": 8}, target=15)

    # Points to predict
    test_points = [{"p1": 2, "p2": 3}, {"p1": 5, "p2": 6}, {"p1": 8, "p2": 9}]

    # Perform predictions
    means = optimizer.predict(test_points)

    # Check that means have correct length
    assert len(means) == len(test_points)

    # Also test with return_std=True to get std
    means, stds = optimizer.predict(test_points, return_std=True)
    assert len(means) == len(test_points)
    assert len(stds) == len(test_points)


def test_predict_example():
    """Test the predict method with known outputs."""
    optimizer = BayesianOptimization(f=target_func, pbounds=PBOUNDS, random_state=1, verbose=0)

    # Register some points
    optimizer.register(params={"p1": 0, "p2": 0}, target=0)
    optimizer.register(params={"p1": 10, "p2": 10}, target=20)

    # Point to predict
    test_point = {"p1": 0, "p2": 0}

    # Perform prediction
    means = optimizer.predict([test_point])
    assert np.isclose(means, 0, atol=1e-3)

    # Test with return_std=True
    means, stds = optimizer.predict([test_point], return_std=True)
    assert np.isclose(means, 0, atol=1e-3)
    assert stds < 0.02  # std should be small but not tiny due to GP uncertainty

    test_point = {"p1": 10, "p2": 10}
    means = optimizer.predict([test_point])
    assert np.isclose(means, 20, atol=1e-3)

    means, stds = optimizer.predict([test_point], return_std=True)
    assert np.isclose(means, 20, atol=1e-3)
    assert stds < 0.02


def test_predict_no_fit():
    """Test the predict method when GP has not been fitted."""
    optimizer = BayesianOptimization(f=target_func, pbounds=PBOUNDS, random_state=1, verbose=0)

    # Perform prediction with fit_gp=True should raise error when no data
    with pytest.raises(RuntimeError):
        optimizer.predict({"p1": 5, "p2": 5}, fit_gp=True)

    # Predict without fitting GP using single dict - get scalar mean by default
    mean = optimizer.predict({"p1": 5, "p2": 5}, fit_gp=False)
    # Since GP is not fitted, mean should be close to 0
    assert np.isclose(mean, 0, atol=1e-4)

    # Get std when not fitting GP
    mean, std = optimizer.predict({"p1": 5, "p2": 5}, fit_gp=False, return_std=True)
    # Since GP is not fitted, std should be large
    assert std > 1e-2

    # Test with list - returns array
    means = optimizer.predict([{"p1": 5, "p2": 5}], fit_gp=False)
    # With a list, even single point returns array
    assert len(means) == 1
    assert np.isclose(means[0], 0, atol=1e-4)


def test_predict_return_cov():
    """Test the predict method with return_cov=True."""
    optimizer = BayesianOptimization(f=target_func, pbounds=PBOUNDS, random_state=1, verbose=0)

    optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    optimizer.register(params={"p1": 4, "p2": 5}, target=9)

    test_points = [{"p1": 2, "p2": 3}, {"p1": 5, "p2": 6}]

    means, cov = optimizer.predict(test_points, return_cov=True)

    assert len(means) == len(test_points)
    assert cov.shape == (len(test_points), len(test_points))


def test_predict_integer_params():
    """Test the predict method with integer parameters."""
    int_pbounds = {"p1": (0, 10, int), "p2": (0, 10, int)}
    optimizer = BayesianOptimization(f=target_func, pbounds=int_pbounds, random_state=1, verbose=0)
    optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    optimizer.register(params={"p1": 4, "p2": 5}, target=9)
    test_points = [{"p1": 2, "p2": 3}, {"p1": 5, "p2": 6}]
    means = optimizer.predict(test_points)
    assert len(means) == len(test_points)

    # Test with return_std
    means, stds = optimizer.predict(test_points, return_std=True)
    assert len(means) == len(test_points)
    assert len(stds) == len(test_points)

    float_points = [{"p1": 2.7, "p2": 3.3}, {"p1": 5.9, "p2": 6.1}]
    means_float = optimizer.predict(float_points)
    assert len(means_float) == len(float_points)

    means_float, stds_float = optimizer.predict(float_points, return_std=True)
    assert len(means_float) == len(float_points)
    assert len(stds_float) == len(float_points)
    # Check that rounding float inputs gives similar predictions as integer inputs

    for i in range(len(test_points)):
        rounded_point = {k: round(v) for k, v in float_points[i].items()}
        mean_rounded = optimizer.predict([rounded_point])
        assert np.isclose(means_float[i], mean_rounded, atol=1e-1)

    # Also check with std
    for i in range(len(test_points)):
        rounded_point = {k: round(v) for k, v in float_points[i].items()}
        mean_rounded, std_rounded = optimizer.predict([rounded_point], return_std=True)
        assert np.isclose(means_float[i], mean_rounded, atol=1e-1)
        assert np.isclose(stds_float[i], std_rounded, atol=1e-1)


def test_predict_categorical_params():
    """Test the predict method with categorical parameters."""

    def cat_target_func(param1: str, param2: str) -> float:
        value_map = {"low": 1.0, "medium": 2.0, "high": 3.0}
        return value_map[param1] + value_map[param2]

    cat_pbounds = {"param1": ["low", "medium", "high"], "param2": ["low", "medium", "high"]}

    optimizer = BayesianOptimization(f=cat_target_func, pbounds=cat_pbounds, random_state=1, verbose=0)

    optimizer.register(params={"param1": "low", "param2": "low"}, target=2.0)
    optimizer.register(params={"param1": "high", "param2": "high"}, target=6.0)

    test_points = [{"param1": "medium", "param2": "medium"}, {"param1": "low", "param2": "high"}]

    means = optimizer.predict(test_points)

    assert len(means) == len(test_points)
    assert np.isclose(means[0], 4.0, atol=1.0)
    assert np.isclose(means[1], 4.0, atol=1.0)

    # Test with return_std
    means, stds = optimizer.predict(test_points, return_std=True)
    assert len(means) == len(test_points)
    assert len(stds) == len(test_points)
    assert stds[0] > 0.0
    assert stds[1] > 0.0


def test_predict_no_points_registered():
    """Test the predict method when no points have been registered."""
    optimizer = BayesianOptimization(f=target_func, pbounds=PBOUNDS, random_state=1, verbose=0)

    test_points = [{"p1": 2, "p2": 3}, {"p1": 5, "p2": 6}]

    means = optimizer.predict(test_points, fit_gp=False)

    assert len(means) == len(test_points)
    for mean in means:
        assert np.isclose(mean, 0, atol=1e-4)

    # Test with return_std to get uncertainty
    means, stds = optimizer.predict(test_points, fit_gp=False, return_std=True)
    assert len(means) == len(test_points)
    assert len(stds) == len(test_points)
    for std in stds:
        assert std > 1e-2


def test_predict_custom_parameter():
    """Test the predict method with a custom parameter type."""

    param = FixedPerimeterTriangleParameter(
        name="sides", bounds=np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]), perimeter=1.0
    )
    pbounds = {"sides": param}

    optimizer = BayesianOptimization(f=area_of_triangle, pbounds=pbounds, random_state=1, verbose=0)

    optimizer.register(
        params={"sides": np.array([0.3, 0.4, 0.3])}, target=area_of_triangle(np.array([0.3, 0.4, 0.3]))
    )
    optimizer.register(
        params={"sides": np.array([0.2, 0.5, 0.3])}, target=area_of_triangle(np.array([0.2, 0.5, 0.3]))
    )

    test_points = [{"sides": np.array([0.25, 0.5, 0.25])}, {"sides": np.array([0.4, 0.4, 0.2])}]

    means = optimizer.predict(test_points)

    assert len(means) == len(test_points)
    for i, point in enumerate(test_points):
        expected_area = area_of_triangle(point["sides"])
        assert np.isclose(means[i], expected_area, atol=0.1)

    # Test with return_std
    means, stds = optimizer.predict(test_points, return_std=True)
    assert len(means) == len(test_points)
    assert len(stds) == len(test_points)
    for i in range(len(test_points)):
        assert stds[i] > 0.0
