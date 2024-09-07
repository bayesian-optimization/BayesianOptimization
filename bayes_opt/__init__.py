"""Pure Python implementation of bayesian global optimization with gaussian processes."""

from __future__ import annotations

import importlib.metadata

from bayes_opt import acquisition
from bayes_opt.bayesian_optimization import BayesianOptimization, Events
from bayes_opt.constraint import ConstraintModel
from bayes_opt.domain_reduction import SequentialDomainReductionTransformer
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.target_space import TargetSpace

__version__ = importlib.metadata.version("bayesian-optimization")


__all__ = [
    "acquisition",
    "BayesianOptimization",
    "TargetSpace",
    "ConstraintModel",
    "Events",
    "ScreenLogger",
    "JSONLogger",
    "SequentialDomainReductionTransformer",
]
