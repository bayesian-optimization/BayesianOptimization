"""Contains utility functions."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bayes_opt.exception import NotUniqueError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bayes_opt.bayesian_optimization import BayesianOptimization


def load_logs(
    optimizer: BayesianOptimization, logs: str | PathLike[str] | Iterable[str | PathLike[str]]
) -> BayesianOptimization:
    """Load previous ...

    Parameters
    ----------
    optimizer : BayesianOptimizer
        Optimizer the register the previous observations with.

    logs : str or os.PathLike
        File to load the logs from.

    Returns
    -------
    The optimizer with the state loaded.

    """
    if isinstance(logs, (str, PathLike)):
        logs = [logs]

    for log in logs:
        with Path(log).open("r") as j:
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
                        constraint_value=(iteration["constraint"] if optimizer.is_constrained else None),
                    )
                except NotUniqueError:
                    continue

    return optimizer


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
