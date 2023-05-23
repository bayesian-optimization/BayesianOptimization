from typing import Callable
import numpy as np
from sklearn.gaussian_process import kernels
from inspect import signature


def is_numeric(value):
    return type(value) in [float, int, complex]


class BayesParameter():

    def __init__(self, name: str, domain) -> None:
        self.name = name
        self.domain = domain

    @property
    def float_bounds(self):
        pass

    def to_float(self, value) -> np.ndarray:
        pass

    def to_param(self, value):
        pass

    def kernel_transform(self, value):
        pass

    @property
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
        return float(value)

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
        return np.array(
            [self.domain[0] - 0.4999999, self.domain[1] + 0.4999999])

    def to_float(self, value) -> np.ndarray:
        return float(value)

    def to_param(self, value):
        return int(np.round(np.squeeze(value)))

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
        one_hot_index = [i for i, val in enumerate(self.domain) if val==value]
        if len(one_hot_index) != 1:
            raise ValueError
        res[one_hot_index] = 1
        return res.astype(float)

    def to_param(self, value):
        return self.domain[np.argmax(value)]

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
    """https://stackoverflow.com/a/58989918/"""
    def copy(target_fct): 
        target_fct.__signature__ = signature(source_fct)
        return target_fct 
    return copy 