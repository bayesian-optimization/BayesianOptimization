"""Parameter classes for Bayesian optimization."""

from __future__ import annotations

import abc
from collections.abc import Sequence
from inspect import signature
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, Union

import numpy as np
from sklearn.gaussian_process import kernels

from bayes_opt.util import ensure_rng

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    Float = np.floating[Any]
    Int = np.integer[Any]

    FloatBoundsWithoutType = tuple[float, float]
    FloatBoundsWithType = tuple[float, float, type[float]]
    FloatBounds = Union[FloatBoundsWithoutType, FloatBoundsWithType]
    IntBounds = tuple[Union[int, float], Union[int, float], type[int]]
    CategoricalBounds = Sequence[Any]
    Bounds = Union[FloatBounds, IntBounds, CategoricalBounds]
    BoundsMapping = Mapping[str, Bounds]

    # FIXME: categorical parameters can be of any type.
    # This will make static type checking for parameters difficult.
    ParamsType = Union[Mapping[str, Any], Sequence[Any], NDArray[Float]]


def is_numeric(value: Any) -> bool:
    """Check if a value is numeric."""
    return isinstance(value, Number) or (
        isinstance(value, np.generic)
        and (np.isdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.number))
    )


class BayesParameter(abc.ABC):
    """Base class for Bayesian optimization parameters.

    Parameters
    ----------
    name : str
        The name of the parameter.
    """

    def __init__(self, name: str, bounds: NDArray[Any]) -> None:
        self.name = name
        self._bounds = bounds

    @property
    def bounds(self) -> NDArray[Any]:
        """The bounds of the parameter in float space."""
        return self._bounds

    @property
    @abc.abstractmethod
    def is_continuous(self) -> bool:
        """Whether the parameter is continuous."""

    def random_sample(
        self, n_samples: int, random_state: np.random.RandomState | int | None
    ) -> NDArray[Float]:
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
    def to_float(self, value: Any) -> float | NDArray[Float]:
        """Convert a parameter value to a float.

        Parameters
        ----------
        value : Any
            The value to convert, should be the canonical representation of the parameter.
        """

    @abc.abstractmethod
    def to_param(self, value: float | NDArray[Float]) -> Any:
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
    def kernel_transform(self, value: NDArray[Float]) -> NDArray[Float]:
        """Transform a parameter value for use in a kernel.

        Parameters
        ----------
        value : np.ndarray
            The value(s) to transform, should be a float.

        Returns
        -------
        np.ndarray
        """

    def to_string(self, value: Any, str_len: int) -> str:
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
        s = f"{value!r:<{str_len}}"

        if len(s) > str_len:
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

    @property
    def is_continuous(self) -> bool:
        """Whether the parameter is continuous."""
        return True

    def to_float(self, value: float) -> float:
        """Convert a parameter value to a float.

        Parameters
        ----------
        value : Any
            The value to convert, should be the canonical representation of the parameter.
        """
        return value

    def to_param(self, value: float | NDArray[Float]) -> float:
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
        return value.flatten()[0]

    def to_string(self, value: float, str_len: int) -> str:
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
            if "." in s and "e" not in s:
                return s[:str_len]
            return s[: str_len - 3] + "..."
        return s

    def kernel_transform(self, value: NDArray[Float]) -> NDArray[Float]:
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

    def __init__(self, name: str, bounds: tuple[int, int]) -> None:
        super().__init__(name, np.array(bounds))

    @property
    def is_continuous(self) -> bool:
        """Whether the parameter is continuous."""
        return False

    def random_sample(
        self, n_samples: int, random_state: np.random.RandomState | int | None
    ) -> NDArray[Float]:
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

    def to_float(self, value: int | float) -> float:
        """Convert a parameter value to a float.

        Parameters
        ----------
        value : Any
            The value to convert, should be the canonical representation of the parameter.
        """
        return float(value)

    def to_param(self, value: int | float | NDArray[Int] | NDArray[Float]) -> int:
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

    def kernel_transform(self, value: NDArray[Float]) -> NDArray[Float]:
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
        if len(categories) != len(set(categories)):
            msg = "Categories must be unique."
            raise ValueError(msg)
        if len(categories) < 2:
            msg = "At least two categories are required."
            raise ValueError(msg)

        self.categories = categories
        lower = np.zeros(self.dim)
        upper = np.ones(self.dim)
        bounds = np.vstack((lower, upper)).T
        super().__init__(name, bounds)

    @property
    def is_continuous(self) -> bool:
        """Whether the parameter is continuous."""
        return False

    def random_sample(
        self, n_samples: int, random_state: np.random.RandomState | int | None
    ) -> NDArray[Float]:
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
        random_state = ensure_rng(random_state)
        res = random_state.randint(0, len(self.categories), n_samples)
        one_hot = np.zeros((n_samples, len(self.categories)))
        one_hot[np.arange(n_samples), res] = 1
        return one_hot.astype(float)

    def to_float(self, value: Any) -> NDArray[Float]:
        """Convert a parameter value to a float.

        Parameters
        ----------
        value : Any
            The value to convert, should be the canonical representation of the parameter.
        """
        res = np.zeros(len(self.categories))
        one_hot_index = [i for i, val in enumerate(self.categories) if val == value]
        res[one_hot_index] = 1
        return res.astype(float)

    def to_param(self, value: float | NDArray[Float]) -> Any:
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
        return self.categories[int(np.argmax(value))]

    def to_string(self, value: Any, str_len: int) -> str:
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
        if not isinstance(value, str):
            value = repr(value)
        s = f"{value:<{str_len}}"

        if len(s) > str_len:
            return s[: str_len - 3] + "..."
        return s

    def kernel_transform(self, value: NDArray[Float]) -> NDArray[Float]:
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
        res[:, np.argmax(value, axis=1)] = 1
        return res

    @property
    def dim(self) -> int:
        """The dimensionality of the parameter."""
        return len(self.categories)


def wrap_kernel(kernel: kernels.Kernel, transform: Callable[[Any], Any]) -> kernels.Kernel:
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
    kernel_type = type(kernel)

    class WrappedKernel(kernel_type):
        @_copy_signature(getattr(kernel_type.__init__, "deprecated_original", kernel_type.__init__))
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)

        def __call__(self, X: Any, Y: Any = None, eval_gradient: bool = False) -> Any:
            X = transform(X)
            Y = transform(Y) if Y is not None else None
            return super().__call__(X, Y, eval_gradient)

        def __reduce__(self) -> str | tuple[Any, ...]:
            return (wrap_kernel, (kernel, transform))

    return WrappedKernel(**kernel.get_params())


def _copy_signature(source_fct: Callable[..., Any]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Clone a signature from a source function to a target function.

    via
    https://stackoverflow.com/a/58989918/
    """

    def copy(target_fct: Callable[..., Any]) -> Callable[..., Any]:
        target_fct.__signature__ = signature(source_fct)
        return target_fct

    return copy
