import unittest
from bayes_opt.observer import Observable


class TestObserver():
    def __init__(self):
        self.counter = 0

    def update(self, event, instance):
        self.counter += 1

class TestObserverPattern(unittest.TestCase):
    def setUp(self):
        events = ['a', 'b']
        self.observable = Observable(events)
        self.observer = TestObserver()

    def test_get_subscribers(self):
        self.observable.register('a', self.observer)
        self.assertTrue(self.observer in self.observable.get_subscribers('a'))
        self.assertTrue(len(self.observable.get_subscribers('a').keys()) == 1)
        self.assertTrue(len(self.observable.get_subscribers('b').keys()) == 0)

    def test_register(self):
        self.observable.register('a', self.observer)
        self.assertTrue(self.observer in self.observable.get_subscribers('a'))

    def test_unregister(self):
        self.observable.register('a', self.observer)
        self.observable.unregister('a', self.observer)
        self.assertTrue(self.observer not in self.observable.get_subscribers('a'))

    def test_dispatch(self):
        test_observer = TestObserver()
        self.observable.register('b', test_observer)
        self.observable.dispatch('b')
        self.observable.dispatch('b')

        self.assertTrue(test_observer.counter == 2)


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_observer.py
    """
    # unittest.main()
    import pytest
    pytest.main([__file__])
