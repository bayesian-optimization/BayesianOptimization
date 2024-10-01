from __future__ import annotations

import abc
from inspect import signature
from typing import Callable

import numpy as np
from sklearn.gaussian_process import kernels


def is_numeric(value):
    return np.issubdtype(type(value), np.number)


class BayesParameter(abc.ABC):
    def __init__(self, name: str, domain) -> None:
        self.name = name
        self.domain = domain

    @property
    @abc.abstractmethod
    def float_bounds(self):
        pass

    @abc.abstractmethod
    def to_float(self, value) -> np.ndarray:
        pass

    @abc.abstractmethod
    def to_param(self, value):
        pass

    @abc.abstractmethod
    def kernel_transform(self, value):
        pass

    def repr(self, value, str_len) -> str:
        s = value.__repr__()

        if len(s) > str_len:
            if "." in s:
                return s[:str_len]
            return s[: str_len - 3] + "..."
        return s

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        pass


class FloatParameter(BayesParameter):
    def __init__(self, name: str, domain) -> None:
        super().__init__(name, domain)

    @property
    def float_bounds(self):
        return np.array(self.domain)

    def to_float(self, value) -> np.ndarray:
        return value

    def to_param(self, value):
        if isinstance(value, np.ndarray) and value.size != 1:
            raise ValueError("FloatParameter scalars")
        return value.flatten()[0]

    def repr(self, value, str_len) -> str:
        s = f"{value:<{str_len}.{str_len}}"
        if len(s) > str_len:
            if "." in s:
                return s[:str_len]
            return s[: str_len - 3] + "..."
        return s

    def kernel_transform(self, value):
        return value

    @property
    def dim(self) -> int:
        return 1


class IntParameter(BayesParameter):
    def __init__(self, name: str, domain) -> None:
        super().__init__(name, domain)

    @property
    def float_bounds(self):
        # adding/subtracting ~0.5 to achieve uniform probability of integers
        return np.array([self.domain[0] - 0.4999999, self.domain[1] + 0.4999999])

    def to_float(self, value) -> np.ndarray:
        return float(value)

    def to_param(self, value):
        return int(np.round(np.squeeze(value)))

    def repr(self, value, str_len) -> str:
        s = f"{value:<{str_len}}"
        if len(s) > str_len:
            if "." in s:
                return s[:str_len]
            return s[: str_len - 3] + "..."
        return s

    def kernel_transform(self, value):
        return np.round(value)

    @property
    def dim(self) -> int:
        return 1


class CategoricalParameter(BayesParameter):
    def __init__(self, name: str, domain) -> None:
        super().__init__(name, domain)

    @property
    def float_bounds(self):
        # to achieve uniform probability after rounding
        lower = np.zeros(self.dim)
        upper = np.ones(self.dim)
        return np.vstack((lower, upper)).T

    def to_float(self, value) -> np.ndarray:
        res = np.zeros(len(self.domain))
        one_hot_index = [i for i, val in enumerate(self.domain) if val == value]
        if len(one_hot_index) != 1:
            raise ValueError
        res[one_hot_index] = 1
        return res.astype(float)

    def to_param(self, value):
        return self.domain[np.argmax(value)]

    def repr(self, value, str_len) -> str:
        s = f"{value:^{str_len}}"
        if len(s) > str_len:
            return s[: str_len - 3] + "..."
        return s

    def kernel_transform(self, value):
        value = np.atleast_2d(value)
        res = np.zeros(value.shape)
        res[np.argmax(value, axis=0)] = 1
        return res

    @property
    def dim(self) -> int:
        return len(self.domain)


def wrap_kernel(kernel: kernels.Kernel, transform: Callable) -> kernels.Kernel:
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
