"""Contains classes and functions for logging."""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from colorama import Fore, just_fix_windows_console

from bayes_opt.event import Events
from bayes_opt.observer import _Tracker

if TYPE_CHECKING:
    from os import PathLike

    from bayes_opt.bayesian_optimization import BayesianOptimization

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
    _colour_new_max = Fore.MAGENTA
    _colour_regular_message = Fore.RESET
    _colour_reset = Fore.RESET

    def __init__(self, verbose: int = 2, is_constrained: bool = False) -> None:
        self._verbose = verbose
        self._is_constrained = is_constrained
        self._header_length = None
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

    def _format_str(self, str_: str) -> str:
        """Format a str.

        Parameters
        ----------
        str_ : str
            Value to format.

        Returns
        -------
        A stringified, formatted version of `x`.
        """
        s = f"{str_:^{self._default_cell_size}}"
        if len(s) > self._default_cell_size:
            return s[: self._default_cell_size - 3] + "..."
        return s

    def _step(self, instance: BayesianOptimization, colour: str = _colour_regular_message) -> str:
        """Log a step.

        Parameters
        ----------
        instance : bayesian_optimization.BayesianOptimization
            The instance associated with the event.

        colour :
            (Default value = _colour_regular_message, equivalent to Fore.RESET)

        Returns
        -------
        A stringified, formatted version of the most recent optimization step.
        """
        res: dict[str, Any] = instance.res[-1]
        keys: list[str] = instance.space.keys
        # iter, target, allowed [, *params]
        cells: list[str | None] = [None] * (3 + len(keys))

        cells[:2] = self._format_number(self._iterations + 1), self._format_number(res["target"])
        if self._is_constrained:
            cells[2] = self._format_bool(res["allowed"])
        params = res.get("params", {})
        cells[3:] = [
            instance.space._params_config[key].to_string(val, self._default_cell_size)
            for key, val in params.items()
        ]

        return "| " + " | ".join(colour + x + self._colour_reset for x in cells if x is not None) + " |"

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
        keys: list[str] = instance.space.keys
        # iter, target, allowed [, *params]
        cells: list[str | None] = [None] * (3 + len(keys))

        cells[:2] = self._format_str("iter"), self._format_str("target")
        if self._is_constrained:
            cells[2] = self._format_str("allowed")
        cells[3:] = [self._format_str(key) for key in keys]

        line = "| " + " | ".join(x for x in cells if x is not None) + " |"
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

    def update(self, event: str, instance: BayesianOptimization) -> None:
        """Handle incoming events.

        Parameters
        ----------
        event : str
            One of the values associated with `Events.OPTIMIZATION_START`,
            `Events.OPTIMIZATION_STEP` or `Events.OPTIMIZATION_END`.

        instance : bayesian_optimization.BayesianOptimization
            The instance associated with the step.
        """
        line = ""
        if event == Events.OPTIMIZATION_START:
            line = self._header(instance) + "\n"
        elif event == Events.OPTIMIZATION_STEP:
            is_new_max = self._is_new_max(instance)
            if self._verbose != 1 or is_new_max:
                colour = self._colour_new_max if is_new_max else self._colour_regular_message
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
    path : str or os.PathLike
        Path to the file to write to.

    reset : bool
        Whether to overwrite the file if it already exists.

    """

    def __init__(self, path: str | PathLike[str], reset: bool = True):
        self._path = Path(path)
        if reset:
            with suppress(OSError):
                self._path.unlink(missing_ok=True)
        super().__init__()

    def update(self, event: str, instance: BayesianOptimization) -> None:
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
            data["datetime"] = {"datetime": now, "elapsed": time_elapsed, "delta": time_delta}

            if "allowed" in data:  # fix: github.com/fmfn/BayesianOptimization/issues/361
                data["allowed"] = bool(data["allowed"])

            if "constraint" in data and isinstance(data["constraint"], np.ndarray):
                data["constraint"] = data["constraint"].tolist()

            with self._path.open("a") as f:
                f.write(json.dumps(data) + "\n")

        self._update_tracker(event, instance)
