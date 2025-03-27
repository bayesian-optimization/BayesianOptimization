from __future__ import annotations

import numpy as np

from bayes_opt.util import ensure_rng


def test_ensure_rng():
    """Test that ensure_rng properly handles different inputs."""

    # Test with None (should return a new RandomState)
    rng1 = ensure_rng(None)
    assert isinstance(rng1, np.random.RandomState)

    # Test with int (should return a new RandomState seeded with that int)
    rng2 = ensure_rng(123)
    assert isinstance(rng2, np.random.RandomState)

    # Test with RandomState (should return the same RandomState)
    rng3 = np.random.RandomState(456)
    rng4 = ensure_rng(rng3)
    assert rng3 is rng4

    # Test that different seeds produce different random numbers
    rng5 = ensure_rng(1)
    rng6 = ensure_rng(2)
    assert rng5.random() != rng6.random()
