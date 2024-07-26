"""Pure Python implementation of bayesian global optimization with gaussian processes."""

from __future__ import annotations

import importlib.metadata

from bayes_opt import acquisition
from bayes_opt.bayesian_optimization import BayesianOptimization, Events
from bayes_opt.constraint import ConstraintModel
from bayes_opt.domain_reduction import SequentialDomainReductionTransformer
from bayes_opt.logger import JSONLogger, ScreenLogger

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
