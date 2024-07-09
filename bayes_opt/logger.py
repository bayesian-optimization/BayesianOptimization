"""Contains classes and functions for logging."""

from __future__ import annotations

import json
import os
from contextlib import suppress
from typing import TYPE_CHECKING

import numpy as np
from colorama import Fore, just_fix_windows_console

from .event import Events
from .observer import _Tracker

if TYPE_CHECKING:
    from .bayesian_optimization import BayesianOptimization

just_fix_windows_console()


def _get_default_logger(verbose: int, is_constrained: bool) -> ScreenLogger:
    """
    Return the default logger.

    Parameters
    ----------
    verbose : int
        Verbosity level of the logger.

    is_constrained : bool
        Whether the underlying optimizer uses constraints (this requires
        an additional column in the output).

    Returns
    -------
    ScreenLogger
        The default logger.

    """
    return ScreenLogger(verbose=verbose, is_constrained=is_constrained)


class ScreenLogger(_Tracker):
    """Logger that outputs text, e.g. to log to a terminal.

    Parameters
    ----------
    verbose : int
        Verbosity level of the logger.

    is_constrained : bool
        Whether the logger is associated with a constrained optimization
        instance.
    """

    _default_cell_size = 9
    _default_precision = 4

    def __init__(self, verbose: int = 2, is_constrained: bool = False) -> None:
        self._verbose = verbose
        self._is_constrained = is_constrained
        self._header_length = 0
        super().__init__()

    @property
    def verbose(self) -> int:
        """Return the verbosity level."""
        return self._verbose

    @verbose.setter
    def verbose(self, v: int) -> None:
        """Set the verbosity level.

        Parameters
        ----------
        v : int
            New verbosity level of the logger.
        """
        self._verbose = v

    @property
    def is_constrained(self) -> bool:
        """Return whether the logger is constrained."""
        return self._is_constrained

    def _format_number(self, x: float) -> str:
        """Format a number.

        Parameters
        ----------
        x : number
            Value to format.

        Returns
        -------
        A stringified, formatted version of `x`.
        """
        if isinstance(x, int):
            s = f"{x:<{self._default_cell_size}}"
        else:
            s = f"{x:<{self._default_cell_size}.{self._default_precision}}"

        if len(s) > self._default_cell_size:
            if "." in s:
                return s[: self._default_cell_size]
            return s[: self._default_cell_size - 3] + "..."
        return s

    def _format_bool(self, x: bool) -> str:
        """Format a boolean.

        Parameters
        ----------
        x : boolean
            Value to format.

        Returns
        -------
        A stringified, formatted version of `x`.
        """
        x_ = ("T" if x else "F") if self._default_cell_size < 5 else str(x)
        return f"{x_:<{self._default_cell_size}}"

    def _format_key(self, key: str) -> str:
        """Format a key.

        Parameters
        ----------
        key : string
            Value to format.

        Returns
        -------
        A stringified, formatted version of `x`.
        """
        s = f"{key:^{self._default_cell_size}}"
        if len(s) > self._default_cell_size:
            return s[: self._default_cell_size - 3] + "..."
        return s

    def _step(self, instance: BayesianOptimization, colour: str = Fore.BLACK) -> str:
        """Log a step.

        Parameters
        ----------
        instance : bayesian_optimization.BayesianOptimization
            The instance associated with the event.

        colour :
            (Default value = Fore.BLACK)

        Returns
        -------
        A stringified, formatted version of the most recent optimization step.
        """
        res = instance.res[-1]
        if "allowed" not in res:
            error_msg = (
                f"Key 'allowed' not found in {res}. "
                "This might be due to the optimizer not being constrained."
            )
            raise KeyError(error_msg)

        cells = []

        cells.append(self._format_number(self._iterations + 1))
        cells.append(self._format_number(res["target"]))
        if self._is_constrained:
            cells.append(self._format_bool(res["allowed"]))

        for key in instance.space.keys:
            cells.append(self._format_number(res["params"][key]))

        return "| " + " | ".join([colour + cells[i] for i in range(len(cells))]) + " |"

    def _header(self, instance: BayesianOptimization) -> str:
        """Print the header of the log.

        Parameters
        ----------
        instance : bayesian_optimization.BayesianOptimization
            The instance associated with the header.

        Returns
        -------
        A stringified, formatted version of the most header.
        """
        cells = []
        cells.append(self._format_key("iter"))
        cells.append(self._format_key("target"))

        if self._is_constrained:
            cells.append(self._format_key("allowed"))

        for key in instance.space.keys:
            cells.append(self._format_key(key))

        line = "| " + " | ".join(cells) + " |"
        self._header_length = len(line)
        return line + "\n" + ("-" * self._header_length)

    def _is_new_max(self, instance: BayesianOptimization) -> bool:
        """Check if the step to log produced a new maximum.

        Parameters
        ----------
        instance : bayesian_optimization.BayesianOptimization
            The instance associated with the step.

        Returns
        -------
        boolean
        """
        if instance.max is None:
            # During constrained optimization, there might not be a maximum
            # value since the optimizer might've not encountered any points
            # that fulfill the constraints.
            return False
        if self._previous_max is None:
            self._previous_max = instance.max["target"]
        return instance.max["target"] > self._previous_max

    def update(self, event: Events, instance: BayesianOptimization) -> None:
        """Handle incoming events.

        Parameters
        ----------
        event : str
            One of the values associated with `Events.OPTIMIZATION_START`,
            `Events.OPTIMIZATION_STEP` or `Events.OPTIMIZATION_END`.

        instance : bayesian_optimization.BayesianOptimization
            The instance associated with the step.
        """
        if event == Events.OPTIMIZATION_START:
            line = self._header(instance) + "\n"
        elif event == Events.OPTIMIZATION_STEP:
            is_new_max = self._is_new_max(instance)
            if self._verbose == 1 and not is_new_max:
                line = ""
            else:
                colour = Fore.MAGENTA if is_new_max else Fore.BLACK
                line = self._step(instance, colour=colour) + "\n"
        elif event == Events.OPTIMIZATION_END:
            line = "=" * self._header_length + "\n"

        if self._verbose:
            print(line, end="")
        self._update_tracker(event, instance)


class JSONLogger(_Tracker):
    """
    Logger that outputs steps in JSON format.

    The resulting file can be used to restart the optimization from an earlier state.

    Parameters
    ----------
    path : str or bytes or os.PathLike
        Path to the file to write to.

    reset : bool
        Whether to overwrite the file if it already exists.

    """

    def __init__(
        self, path: os.PathLike[str] | os.PathLike[bytes], reset: bool = True
    ) -> None:
        self._path = path
        if reset:
            with suppress(OSError):
                os.remove(self._path)
        super().__init__()

    def update(self, event: Events, instance: BayesianOptimization) -> None:
        """
        Handle incoming events.

        Parameters
        ----------
        event : str
            One of the values associated with `Events.OPTIMIZATION_START`,
            `Events.OPTIMIZATION_STEP` or `Events.OPTIMIZATION_END`.

        instance : bayesian_optimization.BayesianOptimization
            The instance associated with the step.

        """
        if event == Events.OPTIMIZATION_STEP:
            data = dict(instance.res[-1])

            now, time_elapsed, time_delta = self._time_metrics()
            data["datetime"] = {
                "datetime": now,
                "elapsed": time_elapsed,
                "delta": time_delta,
            }

            if (
                "allowed" in data
            ):  # fix: github.com/fmfn/BayesianOptimization/issues/361
                data["allowed"] = bool(data["allowed"])

            if "constraint" in data and isinstance(data["constraint"], np.ndarray):
                data["constraint"] = data["constraint"].tolist()

            with open(self._path, "a") as f:
                f.write(json.dumps(data) + "\n")

        self._update_tracker(event, instance)
