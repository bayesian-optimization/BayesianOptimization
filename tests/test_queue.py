import pytest
from bayes_opt.bayesian_optimization import Queue


def test_add():
    queue = Queue()

    assert len(queue) == 0
    assert queue.empty

    queue.add(1)
    assert len(queue) == 1

    queue.add(1)
    assert len(queue) == 2

    queue.add(2)
    assert len(queue) == 3


def test_queue():

    queue = Queue()

    with pytest.raises(StopIteration):
        next(queue)

    queue.add(1)
    queue.add(2)
    queue.add(3)

    assert len(queue) == 3
    assert not queue.empty

    assert next(queue) == 1
    assert len(queue) == 2

    assert next(queue) == 2
    assert next(queue) == 3
    assert len(queue) == 0



if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_observer.py
    """
    pytest.main([__file__])
