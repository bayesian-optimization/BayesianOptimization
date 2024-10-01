"""Holds the parent class for loggers."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from bayes_opt.event import Events

if TYPE_CHECKING:
    from bayes_opt.bayesian_optimization import BayesianOptimization


class _Tracker:
    """Parent class for ScreenLogger and JSONLogger."""

    def __init__(self) -> None:
        self._iterations = 0

        self._previous_max = None
        self._previous_max_params = None

        self._start_time = None
        self._previous_time = None

    def _update_tracker(self, event: str, instance: BayesianOptimization) -> None:
        """Update the tracker.

        Parameters
        ----------
        event : str
            One of the values associated with `Events.OPTIMIZATION_START`,
            `Events.OPTIMIZATION_STEP` or `Events.OPTIMIZATION_END`.

        instance : bayesian_optimization.BayesianOptimization
            The instance associated with the step.
        """
        if event == Events.OPTIMIZATION_STEP:
            self._iterations += 1

            if instance.max is None:
                return

            current_max = instance.max

            if self._previous_max is None or current_max["target"] > self._previous_max:
                self._previous_max = current_max["target"]
                self._previous_max_params = current_max["params"]

    def _time_metrics(self) -> tuple[str, float, float]:
        """Return time passed since last call."""
        now = datetime.now()  # noqa: DTZ005
        if self._start_time is None:
            self._start_time = now
        if self._previous_time is None:
            self._previous_time = now

        time_elapsed = now - self._start_time
        time_delta = now - self._previous_time

        self._previous_time = now
        return (now.strftime("%Y-%m-%d %H:%M:%S"), time_elapsed.total_seconds(), time_delta.total_seconds())
