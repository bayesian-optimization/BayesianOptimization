"""Contains classes and functions for logging."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

from colorama import Fore, just_fix_windows_console

if TYPE_CHECKING:
    from bayes_opt.parameter import ParameterConfig

just_fix_windows_console()


class ScreenLogger:
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

    def __init__(
        self,
        verbose: int = 2,
        is_constrained: bool = False,
        params_config: Mapping[str, ParameterConfig] | None = None,
    ) -> None:
        self._verbose = verbose
        self._is_constrained = is_constrained
        self._params_config = params_config
        self._header_length = None
        self._iterations = 0
        self._previous_max = None
        self._previous_max_params = None
        self._start_time = None
        self._previous_time = None

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

    @property
    def params_config(self) -> Mapping[str, ParameterConfig] | None:
        """Return the parameter configuration used for formatting."""
        return self._params_config

    @params_config.setter
    def params_config(self, config: Mapping[str, ParameterConfig]) -> None:
        """Set the parameter configuration used for formatting.

        Parameters
        ----------
        config : Mapping[str, ParameterConfig]
            New parameter configuration.
        """
        self._params_config = config

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

    def _print_step(
        self, result: dict[str, Any], keys: list[str], colour: str = _colour_regular_message
    ) -> str:
        """Print a step.

        Parameters
        ----------
        result : dict[str, Any]
            The result dictionary for the most recent step.

        keys : list[str]
            The parameter keys.

        colour : str, optional
            Color to use for the output.
            (Default value = _colour_regular_message, equivalent to Fore.RESET)

        Returns
        -------
        A stringified, formatted version of the most recent optimization step.
        """
        if self._params_config is None:
            err_msg = "Parameter configuration is not set. Call set_params_config before logging."
            raise ValueError(err_msg)

        # iter, target, allowed [, *params]
        cells: list[str | None] = [None] * (3 + len(keys))

        cells[:2] = self._format_number(self._iterations + 1), self._format_number(result["target"])
        if self._is_constrained:
            cells[2] = self._format_bool(result["allowed"])
        params = result.get("params", {})
        cells[3:] = [
            self._params_config[key].to_string(val, self._default_cell_size) for key, val in params.items()
        ]

        return "| " + " | ".join(colour + x + self._colour_reset for x in cells if x is not None) + " |"

    def _print_header(self, keys: list[str]) -> str:
        """Print the header of the log.

        Parameters
        ----------
        keys : list[str]
            The parameter keys.

        Returns
        -------
        A stringified, formatted version of the most header.
        """
        # iter, target, allowed [, *params]
        cells: list[str | None] = [None] * (3 + len(keys))

        cells[:2] = self._format_str("iter"), self._format_str("target")
        if self._is_constrained:
            cells[2] = self._format_str("allowed")
        cells[3:] = [self._format_str(key) for key in keys]

        line = "| " + " | ".join(x for x in cells if x is not None) + " |"
        self._header_length = len(line)
        return line + "\n" + ("-" * self._header_length)

    def _is_new_max(self, current_max: dict[str, Any] | None) -> bool:
        """Check if the step to log produced a new maximum.

        Parameters
        ----------
        current_max : dict[str, Any] | None
            The current maximum target value and its parameters.

        Returns
        -------
        boolean
        """
        if current_max is None:
            # During constrained optimization, there might not be a maximum
            # value since the optimizer might've not encountered any points
            # that fulfill the constraints.
            return False
        if self._previous_max is None:
            self._previous_max = current_max["target"]
        return current_max["target"] > self._previous_max

    def _update_tracker(self, current_max: dict[str, Any] | None) -> None:
        """Update the tracker.

        Parameters
        ----------
        current_max : dict[str, Any] | None
            The current maximum target value and its parameters.
        """
        self._iterations += 1

        if current_max is None:
            return

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

    def log_optimization_start(self, keys: list[str]) -> None:
        """Log the start of the optimization process.

        Parameters
        ----------
        keys : list[str]
            The parameter keys.
        """
        if self._verbose:
            line = self._print_header(keys) + "\n"
            print(line, end="")

    def log_optimization_step(
        self, keys: list[str], result: dict[str, Any], current_max: dict[str, Any] | None
    ) -> None:
        """Log an optimization step.

        Parameters
        ----------
        keys : list[str]
            The parameter keys.

        result : dict[str, Any]
            The result dictionary for the most recent step.

        current_max : dict[str, Any] | None
            The current maximum target value and its parameters.
        """
        is_new_max = self._is_new_max(current_max)
        self._update_tracker(current_max)

        if self._verbose != 1 or is_new_max:
            colour = self._colour_new_max if is_new_max else self._colour_regular_message
            line = self._print_step(result, keys, colour=colour) + "\n"
            if self._verbose:
                print(line, end="")

    def log_optimization_end(self) -> None:
        """Log the end of the optimization process."""
        if self._verbose and self._header_length is not None:
            line = "=" * self._header_length + "\n"
            print(line, end="")
