from .bayesian_optimization import BayesianOptimization, Events
from .domain_reduction import SequentialDomainReductionTransformer
from .util import UtilityFunction
from .logger import ScreenLogger, JSONLogger
from .constraint import ConstraintModel

import importlib.metadata
# __version__ = importlib.metadata.version(__name__)
# nb: this wont work unless the package name (in pyproject.toml) and the package are consistently named.



__all__ = [
    "BayesianOptimization",
    "ConstraintModel",
    "UtilityFunction",
    "Events",
    "ScreenLogger",
    "JSONLogger",
    "SequentialDomainReductionTransformer",
]

