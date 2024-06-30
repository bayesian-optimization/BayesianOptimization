"""Pure Python implementation of bayesian global optimization with gaussian processes."""
from .bayesian_optimization import BayesianOptimization, Events
from .domain_reduction import SequentialDomainReductionTransformer 
from .logger import ScreenLogger, JSONLogger
from .constraint import ConstraintModel
from . import acquisition

import importlib.metadata
__version__ = importlib.metadata.version('bayesian-optimization')



__all__ = [
    "acquisition",
    "BayesianOptimization",
    "ConstraintModel",
    "Events",
    "ScreenLogger",
    "JSONLogger",
    "SequentialDomainReductionTransformer",
]

