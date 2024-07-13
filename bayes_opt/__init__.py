"""Pure Python implementation of bayesian global optimization with gaussian processes."""

from __future__ import annotations

import importlib.metadata

from . import acquisition
from .bayesian_optimization import BayesianOptimization, Events
from .constraint import ConstraintModel
from .domain_reduction import SequentialDomainReductionTransformer
from .logger import JSONLogger, ScreenLogger

__version__ = importlib.metadata.version("bayesian-optimization")


__all__ = [
    "acquisition",
    "BayesianOptimization",
    "ConstraintModel",
    "Events",
    "ScreenLogger",
    "JSONLogger",
    "SequentialDomainReductionTransformer",
]
