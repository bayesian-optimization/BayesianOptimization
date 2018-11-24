from bayes_opt.bayesian_optimization import Observable


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


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_observer.py
    """
    import pytest
    pytest.main([__file__])
