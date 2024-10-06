"""Parameter classes for Bayesian optimization."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from inspect import signature
from typing import Any, Callable

import numpy as np
from sklearn.gaussian_process import kernels

from bayes_opt.util import ensure_rng


def is_numeric(value):
    """Check if a value is numeric."""
    return np.issubdtype(type(value), np.number)


class BayesParameter(abc.ABC):
    """Base class for Bayesian optimization parameters.

    Parameters
    ----------
    name : str
        The name of the parameter.
    """

    def __init__(self, name: str, bounds) -> None:
        self.name = name
        self._bounds = bounds

    @property
    def bounds(self):
        """The bounds of the parameter in float space."""
        return self._bounds

    def random_sample(self, n_samples: int, random_state: np.random.RandomState | int | None) -> np.ndarray:
        """Generate random samples from the parameter.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.

        random_state : np.random.RandomState | int | None
            The random state to use for sampling.

        Returns
        -------
        np.ndarray
            The samples.
        """
        random_state = ensure_rng(random_state)
        return random_state.uniform(self.bounds[0], self.bounds[1], n_samples)

    @abc.abstractmethod
    def to_float(self, value) -> np.ndarray:
        """Convert a parameter value to a float.

        Parameters
        ----------
        value : Any
            The value to convert, should be the canonical representation of the parameter.
        """

    @abc.abstractmethod
    def to_param(self, value):
        """Convert a float value to a parameter.

        Parameters
        ----------
        value : np.ndarray
            The value to convert, should be a float.

        Returns
        -------
        Any
            The canonical representation of the parameter.
        """

    @abc.abstractmethod
    def kernel_transform(self, value):
        """Transform a parameter value for use in a kernel.

        Parameters
        ----------
        value : np.ndarray
            The value(s) to transform, should be a float.

        Returns
        -------
        np.ndarray
        """

    def repr(self, value, str_len) -> str:
        """Represent a parameter value as a string.

        Parameters
        ----------
        value : Any
            The value to represent.

        str_len : int
            The maximum length of the string representation.

        Returns
        -------
        str
        """
        s = value.__repr__()

        if len(s) > str_len:
            if "." in s:
                return s[:str_len]
            return s[: str_len - 3] + "..."
        return s

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """The dimensionality of the parameter."""


class FloatParameter(BayesParameter):
    """A parameter with float values.

    Parameters
    ----------
    name : str
        The name of the parameter.

    bounds : tuple[float, float]
        The bounds of the parameter.
    """

    def __init__(self, name: str, bounds: tuple[float, float]) -> None:
        super().__init__(name, np.array(bounds))

    def to_float(self, value) -> np.ndarray:
        """Convert a parameter value to a float.

        Parameters
        ----------
        value : Any
            The value to convert, should be the canonical representation of the parameter.
        """
        return value

    def to_param(self, value):
        """Convert a float value to a parameter.

        Parameters
        ----------
        value : np.ndarray
            The value to convert, should be a float.

        Returns
        -------
        Any
            The canonical representation of the parameter.
        """
        if isinstance(value, np.ndarray) and value.size != 1:
            msg = "FloatParameter value should be scalar"
            raise ValueError(msg)
        return value.flatten()[0]

    def repr(self, value, str_len) -> str:
        """Represent a parameter value as a string.

        Parameters
        ----------
        value : Any
            The value to represent.

        str_len : int
            The maximum length of the string representation.

        Returns
        -------
        str
        """
        s = f"{value:<{str_len}.{str_len}}"
        if len(s) > str_len:
            if "." in s:
                return s[:str_len]
            return s[: str_len - 3] + "..."
        return s

    def kernel_transform(self, value):
        """Transform a parameter value for use in a kernel.

        Parameters
        ----------
        value : np.ndarray
            The value(s) to transform, should be a float.

        Returns
        -------
        np.ndarray
        """
        return value

    @property
    def dim(self) -> int:
        """The dimensionality of the parameter."""
        return 1


class IntParameter(BayesParameter):
    """A parameter with int values.

    Parameters
    ----------
    name : str
        The name of the parameter.

    bounds : tuple[int, int]
        The bounds of the parameter.
    """

    def __init__(self, name: str, bounds: tuple[int | float, int | float]) -> None:
        super().__init__(name, np.array(bounds))

    def random_sample(self, n_samples: int, random_state: np.random.RandomState | int | None) -> np.ndarray:
        """Generate random samples from the parameter.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.

        random_state : np.random.RandomState | int | None
            The random state to use for sampling.

        Returns
        -------
        np.ndarray
            The samples.
        """
        random_state = ensure_rng(random_state)
        return random_state.randint(self.bounds[0], self.bounds[1] + 1, n_samples).astype(float)

    def to_float(self, value) -> np.ndarray:
        """Convert a parameter value to a float.

        Parameters
        ----------
        value : Any
            The value to convert, should be the canonical representation of the parameter.
        """
        return float(value)

    def to_param(self, value):
        """Convert a float value to a parameter.

        Parameters
        ----------
        value : np.ndarray
            The value to convert, should be a float.

        Returns
        -------
        Any
            The canonical representation of the parameter.
        """
        return int(np.round(np.squeeze(value)))

    def repr(self, value, str_len) -> str:
        """Represent a parameter value as a string.

        Parameters
        ----------
        value : Any
            The value to represent.

        str_len : int
            The maximum length of the string representation.

        Returns
        -------
        str
        """
        s = f"{value:<{str_len}}"
        if len(s) > str_len:
            if "." in s:
                return s[:str_len]
            return s[: str_len - 3] + "..."
        return s

    def kernel_transform(self, value):
        """Transform a parameter value for use in a kernel.

        Parameters
        ----------
        value : np.ndarray
            The value(s) to transform, should be a float.

        Returns
        -------
        np.ndarray
        """
        return np.round(value)

    @property
    def dim(self) -> int:
        """The dimensionality of the parameter."""
        return 1


class CategoricalParameter(BayesParameter):
    """A parameter with categorical values.

    Parameters
    ----------
    name : str
        The name of the parameter.

    categories : Sequence[Any]
        The categories of the parameter.
    """

    def __init__(self, name: str, categories: Sequence[Any]) -> None:
        self.categories = categories
        lower = np.zeros(self.dim)
        upper = np.ones(self.dim)
        bounds = np.vstack((lower, upper)).T
        super().__init__(name, bounds)

    def random_sample(self, n_samples: int, random_state: np.random.RandomState | int | None) -> np.ndarray:
        """Generate random float-format samples from the parameter.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.

        random_state : np.random.RandomState | int | None
            The random state to use for sampling.

        Returns
        -------
        np.ndarray
            The samples.
        """
        res = random_state.randint(0, len(self.categories), n_samples)
        one_hot = np.zeros((n_samples, len(self.categories)))
        one_hot[np.arange(n_samples), res] = 1
        return one_hot.astype(float)

    def to_float(self, value) -> np.ndarray:
        """Convert a parameter value to a float.

        Parameters
        ----------
        value : Any
            The value to convert, should be the canonical representation of the parameter.
        """
        res = np.zeros(len(self.categories))
        one_hot_index = [i for i, val in enumerate(self.categories) if val == value]
        if len(one_hot_index) != 1:
            raise ValueError
        res[one_hot_index] = 1
        return res.astype(float)

    def to_param(self, value):
        """Convert a float value to a parameter.

        Parameters
        ----------
        value : np.ndarray
            The value to convert, should be a float.

        Returns
        -------
        Any
            The canonical representation of the parameter.
        """
        return self.categories[np.argmax(value)]

    def repr(self, value, str_len) -> str:
        """Represent a parameter value as a string.

        Parameters
        ----------
        value : Any
            The value to represent.

        str_len : int
            The maximum length of the string representation.

        Returns
        -------
        str
        """
        s = f"{value:^{str_len}}"
        if len(s) > str_len:
            return s[: str_len - 3] + "..."
        return s

    def kernel_transform(self, value):
        """Transform a parameter value for use in a kernel.

        Parameters
        ----------
        value : np.ndarray
            The value(s) to transform, should be a float.

        Returns
        -------
        np.ndarray
        """
        value = np.atleast_2d(value)
        res = np.zeros(value.shape)
        res[np.argmax(value, axis=0)] = 1
        return res

    @property
    def dim(self) -> int:
        """The dimensionality of the parameter."""
        return len(self.categories)


def wrap_kernel(kernel: kernels.Kernel, transform: Callable) -> kernels.Kernel:
    """Wrap a kernel to transform input data before passing it to the kernel.

    Parameters
    ----------
    kernel : kernels.Kernel
        The kernel to wrap.

    transform : Callable
        The transformation function to apply to the input data.

    Returns
    -------
    kernels.Kernel
        The wrapped kernel.

    Notes
    -----
    See https://arxiv.org/abs/1805.03463 for more information.
    """

    class WrappedKernel(type(kernel)):
        @copy_signature(getattr(kernel.__class__.__init__, "deprecated_original", kernel.__class__.__init__))
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)

        def __call__(self, X, Y=None, eval_gradient=False):
            X = transform(X)
            return super().__call__(X, Y, eval_gradient)

    return WrappedKernel(**kernel.get_params())


def copy_signature(source_fct):
    """Clones a signature from a source function to a target function.

    via
    https://stackoverflow.com/a/58989918/
    """

    def copy(target_fct):
        target_fct.__signature__ = signature(source_fct)
        return target_fct

    return copy
