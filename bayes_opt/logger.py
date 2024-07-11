"""Contains classes and functions for logging."""
from __future__ import print_function
import os
import json
import numpy as np
from .observer import _Tracker
from .event import Events
from colorama import Fore, just_fix_windows_console

just_fix_windows_console()

def _get_default_logger(verbose, is_constrained):
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
    
    def __init__(self, verbose=2, is_constrained=False):
        self._verbose = verbose
        self._is_constrained = is_constrained
        self._header_length = None
        super().__init__()

    @property
    def verbose(self):
        """Return the verbosity level."""
        return self._verbose

    @verbose.setter
    def verbose(self, v):
        """Set the verbosity level.

        Parameters
        ----------
        v : int
            New verbosity level of the logger.
        """
        self._verbose = v

    @property
    def is_constrained(self):
        """Return whether the logger is constrained."""
        return self._is_constrained

    def _format_number(self, x):
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
                return s[:self._default_cell_size]
            return s[:self._default_cell_size - 3] + "..."
        return s

    def _format_bool(self, x):
        """Format a boolean.

        Parameters
        ----------
        x : boolean
            Value to format.

        Returns
        -------
        A stringified, formatted version of `x`.
        """
        if 5 > self._default_cell_size:
            if x:
                x_ = 'T'
            else:
                x_ = 'F'
        else:
            x_ = str(x)
        s = f"{x_:<{self._default_cell_size}}"
        return s

    def _format_key(self, key):
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
            return s[:self._default_cell_size - 3] + "..."
        return s

    def _step(self, instance, colour=_colour_regular_message):
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
        res = instance.res[-1]
        cells = []

        cells.append(self._format_number(self._iterations + 1))
        cells.append(self._format_number(res["target"]))
        if self._is_constrained:
            cells.append(self._format_bool(res["allowed"]))


        for key in instance.space.keys:
            cells.append(self._format_number(res["params"][key]))

        return "| " + " | ".join([colour + cells[i] + self._colour_reset for i in range(len(cells))]) + " |"

    def _header(self, instance):
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

    def _is_new_max(self, instance):
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

    def update(self, event, instance):
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
    path : str or bytes or os.PathLike
        Path to the file to write to.

    reset : bool
        Whether to overwrite the file if it already exists.

    """

    def __init__(self, path, reset=True):
        self._path = path
        if reset:
            try:
                os.remove(self._path)
            except OSError:
                pass
        super().__init__()

    def update(self, event, instance):
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

            if "allowed" in data: # fix: github.com/fmfn/BayesianOptimization/issues/361
                data["allowed"] = bool(data["allowed"])
            
            if "constraint" in data and isinstance(data["constraint"], np.ndarray):
                data["constraint"] = data["constraint"].tolist()

            with open(self._path, "a") as f:
                f.write(json.dumps(data) + "\n")

        self._update_tracker(event, instance)
