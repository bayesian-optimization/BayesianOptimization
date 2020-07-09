import pytest
import numpy as np
from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization
from bayes_opt.logger import ScreenLogger
from bayes_opt.event import Events, DEFAULT_EVENTS


def target_func(**kwargs):
    # arbitrary target func
    return sum(kwargs.values())


PBOUNDS = {'p1': (0, 10), 'p2': (0, 10)}


def test_register():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    assert len(optimizer.space) == 0

    optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    assert len(optimizer.res) == 1
    assert len(optimizer.space) == 1

    optimizer.space.register(params={"p1": 5, "p2": 4}, target=9)
    assert len(optimizer.res) == 2
    assert len(optimizer.space) == 2

    with pytest.raises(KeyError):
        optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    with pytest.raises(KeyError):
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
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

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
    assert len(optimizer.space) == 2
    assert len(optimizer._queue) == 0
    assert optimizer.max["target"] == 6
    assert optimizer.max["params"] == {"p1": 3, "p2": 3}


def test_suggest_at_random():
    util = UtilityFunction(kind="poi", kappa=5, xi=0)
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    for _ in range(50):
        sample = optimizer.space.params_to_array(optimizer.suggest(util))
        assert len(sample) == optimizer.space.dim
        assert all(sample >= optimizer.space.bounds[:, 0])
        assert all(sample <= optimizer.space.bounds[:, 1])


def test_suggest_with_one_observation():
    util = UtilityFunction(kind="ucb", kappa=5, xi=0)
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    optimizer.register(params={"p1": 1, "p2": 2}, target=3)

    for _ in range(5):
        sample = optimizer.space.params_to_array(optimizer.suggest(util))
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


def test_prime_subscriptions():
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    optimizer._prime_subscriptions()

    # Test that the default observer is correctly subscribed
    for event in DEFAULT_EVENTS:
        assert all([
            isinstance(k, ScreenLogger) for k in
            optimizer._events[event].keys()
        ])
        assert all([
            hasattr(k, "update") for k in
            optimizer._events[event].keys()
        ])

    test_subscriber = "test_subscriber"

    def test_callback(event, instance):
        pass

    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    optimizer.subscribe(
        event=Events.OPTIMIZATION_START,
        subscriber=test_subscriber,
        callback=test_callback,
    )
    # Test that the desired observer is subscribed
    assert all([
        k == test_subscriber for k in
        optimizer._events[Events.OPTIMIZATION_START].keys()
    ])
    assert all([
        v == test_callback for v in
        optimizer._events[Events.OPTIMIZATION_START].values()
    ])

    # Check that prime subscriptions won't overight manual subscriptions
    optimizer._prime_subscriptions()
    assert all([
        k == test_subscriber for k in
        optimizer._events[Events.OPTIMIZATION_START].keys()
    ])
    assert all([
        v == test_callback for v in
        optimizer._events[Events.OPTIMIZATION_START].values()
    ])

    assert optimizer._events[Events.OPTIMIZATION_STEP] == {}
    assert optimizer._events[Events.OPTIMIZATION_END] == {}

    with pytest.raises(KeyError):
        optimizer._events["other"]


def test_set_bounds():
    pbounds = {
        'p1': (0, 1),
        'p3': (0, 3),
        'p2': (0, 2),
        'p4': (0, 4),
    }
    optimizer = BayesianOptimization(target_func, pbounds, random_state=1)

    # Ignore unknown keys
    optimizer.set_bounds({"other": (7, 8)})
    assert all(optimizer.space.bounds[:, 0] == np.array([0, 0, 0, 0]))
    assert all(optimizer.space.bounds[:, 1] == np.array([1, 2, 3, 4]))

    # Update bounds accordingly
    optimizer.set_bounds({"p2": (1, 8)})
    assert all(optimizer.space.bounds[:, 0] == np.array([0, 1, 0, 0]))
    assert all(optimizer.space.bounds[:, 1] == np.array([1, 8, 3, 4]))


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
    from sklearn.exceptions import NotFittedError
    class Tracker:
        def __init__(self):
            self.start_count = 0
            self.step_count = 0
            self.end_count = 0

        def update_start(self, event, instance):
            self.start_count += 1

        def update_step(self, event, instance):
            self.step_count += 1

        def update_end(self, event, instance):
            self.end_count += 1

        def reset(self):
            self.__init__()

    optimizer = BayesianOptimization(target_func, PBOUNDS,
                                     random_state=np.random.RandomState(1))

    tracker = Tracker()
    optimizer.subscribe(
        event=Events.OPTIMIZATION_START,
        subscriber=tracker,
        callback=tracker.update_start,
    )
    optimizer.subscribe(
        event=Events.OPTIMIZATION_STEP,
        subscriber=tracker,
        callback=tracker.update_step,
    )
    optimizer.subscribe(
        event=Events.OPTIMIZATION_END,
        subscriber=tracker,
        callback=tracker.update_end,
    )

    optimizer.maximize(init_points=0, n_iter=0)
    assert optimizer._queue.empty
    assert len(optimizer.space) == 1
    assert tracker.start_count == 1
    assert tracker.step_count == 1
    assert tracker.end_count == 1

    optimizer.maximize(init_points=2, n_iter=0, alpha=1e-2)
    assert optimizer._queue.empty
    assert len(optimizer.space) == 3
    assert optimizer._gp.alpha == 1e-2
    assert tracker.start_count == 2
    assert tracker.step_count == 3
    assert tracker.end_count == 2

    optimizer.maximize(init_points=0, n_iter=2)
    assert optimizer._queue.empty
    assert len(optimizer.space) == 5
    assert tracker.start_count == 3
    assert tracker.step_count == 5
    assert tracker.end_count == 3


def test_define_wrong_transformer():
    with pytest.raises(TypeError):
        optimizer = BayesianOptimization(target_func, PBOUNDS,
                                         random_state=np.random.RandomState(1),
                                         bounds_transformer=3)


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_bayesian_optimization.py
    """
    pytest.main([__file__])
