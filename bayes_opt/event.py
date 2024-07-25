"""Register optimization events variables."""

from __future__ import annotations

from enum import Enum

__all__ = ["Events", "DEFAULT_EVENTS"]


class Events(Enum):
    """Define optimization events.

    Behaves similar to enums.
    """

    OPTIMIZATION_START = "optimization:start"
    OPTIMIZATION_STEP = "optimization:step"
    OPTIMIZATION_END = "optimization:end"


DEFAULT_EVENTS: tuple[Events, ...] = tuple(Events)
