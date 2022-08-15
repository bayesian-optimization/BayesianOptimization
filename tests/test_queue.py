import pytest
from queue import Queue, Empty


def test_add():
    queue = Queue()

    assert queue.empty()

    queue.put(1)
    assert queue.qsize() == 1

    queue.put(1)
    assert queue.qsize() == 2

    queue.put(2)
    assert queue.qsize() == 3


def test_queue():

    queue = Queue()

    with pytest.raises(Empty):
        queue.get(block=False)

    queue.put(1)
    queue.put(2)
    queue.put(3)

    assert queue.qsize() == 3
    assert not queue.empty()

    assert queue.get() == 1
    assert queue.qsize() == 2

    assert queue.get() == 2
    assert queue.get() == 3
    assert queue.empty()



if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_observer.py
    """
    pytest.main([__file__])
