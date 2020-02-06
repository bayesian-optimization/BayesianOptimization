from bayes_opt.bayesian_optimization import Observable
from bayes_opt.observer import _Tracker
from bayes_opt.event import Events


EVENTS = ["a", "b", "c"]


class SimpleObserver():
    def __init__(self):
        self.counter = 0

    def update(self, event, instance):
        self.counter += 1


def test_get_subscribers():
    observer = SimpleObserver()
    observable = Observable(events=EVENTS)
    observable.subscribe("a", observer)

    assert observer in observable.get_subscribers('a')
    assert observer not in observable.get_subscribers('b')
    assert observer not in observable.get_subscribers('c')

    assert len(observable.get_subscribers('a')) == 1
    assert len(observable.get_subscribers('b')) == 0
    assert len(observable.get_subscribers('c')) == 0


def test_unsubscribe():
    observer = SimpleObserver()
    observable = Observable(events=EVENTS)

    observable.subscribe("a", observer)
    observable.unsubscribe("a", observer)

    assert observer not in observable.get_subscribers('a')
    assert len(observable.get_subscribers('a')) == 0


def test_dispatch():
    observer_a = SimpleObserver()
    observer_b = SimpleObserver()
    observable = Observable(events=EVENTS)

    observable.subscribe("a", observer_a)
    observable.subscribe("b", observer_b)

    assert observer_a.counter == 0
    assert observer_b.counter == 0

    observable.dispatch('b')
    assert observer_a.counter == 0
    assert observer_b.counter == 1

    observable.dispatch('a')
    observable.dispatch('b')
    assert observer_a.counter == 1
    assert observer_b.counter == 2

    observable.dispatch('a')
    observable.dispatch('c')
    assert observer_a.counter == 2
    assert observer_a.counter == 2


def test_tracker():
    class MockInstance:
        def __init__(self, max_target=1, max_params=[1, 1]):
            self._max_target = max_target
            self._max_params = max_params

        @property
        def max(self):
            return {"target": self._max_target, "params": self._max_params}

    tracker = _Tracker()
    assert tracker._iterations == 0
    assert tracker._previous_max is None
    assert tracker._previous_max_params is None

    test_instance = MockInstance()
    tracker._update_tracker("other_event", test_instance)
    assert tracker._iterations == 0
    assert tracker._previous_max is None
    assert tracker._previous_max_params is None

    tracker._update_tracker(Events.OPTIMIZATION_STEP, test_instance)
    assert tracker._iterations == 1
    assert tracker._previous_max == 1
    assert tracker._previous_max_params == [1, 1]

    new_instance = MockInstance(max_target=7, max_params=[7, 7])
    tracker._update_tracker(Events.OPTIMIZATION_STEP, new_instance)
    assert tracker._iterations == 2
    assert tracker._previous_max == 7
    assert tracker._previous_max_params == [7, 7]

    other_instance = MockInstance(max_target=2, max_params=[2, 2])
    tracker._update_tracker(Events.OPTIMIZATION_STEP, other_instance)
    assert tracker._iterations == 3
    assert tracker._previous_max == 7
    assert tracker._previous_max_params == [7, 7]

    tracker._time_metrics()
    start_time = tracker._start_time
    previous_time = tracker._previous_time

    tracker._time_metrics()
    assert start_time == tracker._start_time
    assert previous_time < tracker._previous_time


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_observer.py
    """
    import pytest
    pytest.main([__file__])
