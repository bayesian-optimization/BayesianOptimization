"""Pure Python implementation of bayesian global optimization with gaussian processes."""
from .bayesian_optimization import BayesianOptimization, Events
from .domain_reduction import SequentialDomainReductionTransformer
from .util import UtilityFunction
from .logger import ScreenLogger, JSONLogger
from .constraint import ConstraintModel
from .util import UtilityFunction

import importlib.metadata
__version__ = importlib.metadata.version('bayesian-optimization')



__all__ = [
    "BayesianOptimization",
    "ConstraintModel",
    "UtilityFunction",
    "Events",
    "ScreenLogger",
    "JSONLogger",
    "SequentialDomainReductionTransformer",
]

