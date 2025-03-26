"""Contains utility functions."""

from __future__ import annotations

import numpy as np


def ensure_rng(random_state: int | np.random.RandomState | None = None) -> np.random.RandomState:
    """Create a random number generator based on an optional seed.

    Parameters
    ----------
    random_state : np.random.RandomState or int or None, default=None
        Random state to use. if `None`, will create an unseeded random state.
        If `int`, creates a state using the argument as seed. If a
        `np.random.RandomState` simply returns the argument.

    Returns
    -------
    np.random.RandomState

    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    elif not isinstance(random_state, np.random.RandomState):
        error_msg = "random_state should be an instance of np.random.RandomState, an int, or None."
        raise TypeError(error_msg)
    return random_state
