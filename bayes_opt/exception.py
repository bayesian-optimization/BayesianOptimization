"""This module contains custom exceptions for Bayesian Optimization."""

from __future__ import annotations

__all__ = [
    "BayesianOptimizationError",
    "ConstraintNotSupportedError",
    "NoValidPointRegisteredError",
    "NotUniqueError",
    "TargetSpaceEmptyError",
]


class BayesianOptimizationError(Exception):
    """Base class for exceptions in the Bayesian Optimization."""


class NotUniqueError(BayesianOptimizationError):
    """A point is non-unique."""


class ConstraintNotSupportedError(BayesianOptimizationError):
    """Raised when constrained optimization is not supported."""


class NoValidPointRegisteredError(BayesianOptimizationError):
    """Raised when an acquisition function depends on previous points but none are registered."""


class TargetSpaceEmptyError(BayesianOptimizationError):
    """Raised when the target space is empty."""
