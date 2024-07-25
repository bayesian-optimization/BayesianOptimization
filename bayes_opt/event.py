"""Register optimization events variables."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        __slots__ = ()

        def __str__(self) -> str:
            return str(self.value)


__all__ = ["Events", "DEFAULT_EVENTS"]


class Events(StrEnum):
    """Define optimization events.

    Behaves similar to enums.
    """

    OPTIMIZATION_START = "optimization:start"
    OPTIMIZATION_STEP = "optimization:step"
    OPTIMIZATION_END = "optimization:end"


DEFAULT_EVENTS: frozenset[Events] = frozenset(Events)
