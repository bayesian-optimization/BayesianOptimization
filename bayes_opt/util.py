"""Contains utility functions."""
import json
import numpy as np


class NotUniqueError(Exception):
    """A point is non-unique."""


def load_logs(optimizer, logs):
    """Load previous ...

    Parameters
    ----------
    optimizer : BayesianOptimizer
        Optimizer the register the previous observations with.

    logs : str or bytes or os.PathLike
        File to load the logs from.

    Returns
    -------
    The optimizer with the state loaded.
    
    """
    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"],
                        constraint_value=(
                            iteration["constraint"]
                            if optimizer.is_constrained else None
                        )
                    )
                except NotUniqueError:
                    continue

    return optimizer


def ensure_rng(random_state=None):
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
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state

