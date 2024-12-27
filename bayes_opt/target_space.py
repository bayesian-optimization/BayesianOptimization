"""Manages the optimization domain and holds points."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np
from colorama import Fore

from bayes_opt.exception import NotUniqueError
from bayes_opt.parameter import BayesParameter, CategoricalParameter, FloatParameter, IntParameter, is_numeric
from bayes_opt.util import ensure_rng

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.random import RandomState
    from numpy.typing import NDArray

    from bayes_opt.constraint import ConstraintModel
    from bayes_opt.parameter import BoundsMapping, ParamsType

    Float = np.floating[Any]
    Int = np.integer[Any]


def _hashable(x: NDArray[Float]) -> tuple[float, ...]:
    """Ensure that a point is hashable by a python dict."""
    return tuple(map(float, x))


class TargetSpace:
    """Holds the param-space coordinates (X) and target values (Y).

    Allows for constant-time appends.

    Parameters
    ----------
    target_func : function or None.
        Function to be maximized.

    pbounds : dict
        Dictionary with parameters names as keys and a tuple with minimum
        and maximum values.

    random_state : int, RandomState, or None
        optionally specify a seed for a random number generator

    allow_duplicate_points: bool, optional (default=False)
        If True, the optimizer will allow duplicate points to be registered.
        This behavior may be desired in high noise situations where repeatedly probing
        the same point will give different answers. In other situations, the acquisition
        may occasionally generate a duplicate point.

    Examples
    --------
    >>> def target_func(p1, p2):
    >>>     return p1 + p2
    >>> pbounds = {"p1": (0, 1), "p2": (1, 100)}
    >>> space = TargetSpace(target_func, pbounds, random_state=0)
    >>> x = np.array([4, 5])
    >>> y = target_func(x)
    >>> space.register(x, y)
    >>> assert self.max()["target"] == 9
    >>> assert self.max()["params"] == {"p1": 1.0, "p2": 2.0}
    """

    def __init__(
        self,
        target_func: Callable[..., float] | None,
        pbounds: BoundsMapping,
        constraint: ConstraintModel | None = None,
        random_state: int | RandomState | None = None,
        allow_duplicate_points: bool | None = False,
    ) -> None:
        self.random_state = ensure_rng(random_state)
        self._allow_duplicate_points = allow_duplicate_points or False
        self.n_duplicate_points = 0

        # The function to be optimized
        self.target_func = target_func

        # Get the name of the parameters
        self._keys: list[str] = list(pbounds.keys())

        self._params_config = self.make_params(pbounds)
        self._dim = sum([self._params_config[key].dim for key in self._keys])

        self._masks = self.make_masks()
        self._bounds = self.calculate_bounds()

        # preallocated memory for X and Y points
        self._params: NDArray[Float] = np.empty(shape=(0, self.dim))
        self._target: NDArray[Float] = np.empty(shape=(0,))

        # keep track of unique points we have seen so far
        self._cache: dict[tuple[float, ...], float | tuple[float, float | NDArray[Float]]] = {}

        self._constraint: ConstraintModel | None = constraint

        if constraint is not None:
            # preallocated memory for constraint fulfillment
            self._constraint_values: NDArray[Float]
            if constraint.lb.size == 1:
                self._constraint_values = np.empty(shape=(0), dtype=float)
            else:
                self._constraint_values = np.empty(shape=(0, self._constraint.lb.size), dtype=float)
        else:
            self._constraint = None

    def __contains__(self, x: NDArray[Float]) -> bool:
        """Check if this parameter has already been registered.

        Returns
        -------
        bool
        """
        return _hashable(x) in self._cache

    def __len__(self) -> int:
        """Return number of observations registered.

        Returns
        -------
        int
        """
        return len(self._target)

    @property
    def empty(self) -> bool:
        """Check if anything has been registered.

        Returns
        -------
        bool
        """
        return len(self) == 0

    @property
    def params(self) -> NDArray[Float]:
        """Get the parameter values registered to this TargetSpace.

        Returns
        -------
        np.ndarray
        """
        return self._params

    @property
    def target(self) -> NDArray[Float]:
        """Get the target function values registered to this TargetSpace.

        Returns
        -------
        np.ndarray
        """
        return self._target

    @property
    def dim(self) -> int:
        """Get the number of parameter names.

        Returns
        -------
        int
        """
        return self._dim

    @property
    def keys(self) -> list[str]:
        """Get the keys (or parameter names).

        Returns
        -------
        list of str
        """
        return self._keys

    @property
    def params_config(self) -> dict[str, BayesParameter]:
        """Get the parameters configuration."""
        return self._params_config

    @property
    def bounds(self) -> NDArray[Float]:
        """Get the bounds of this TargetSpace.

        Returns
        -------
        np.ndarray
        """
        return self._bounds

    @property
    def constraint(self) -> ConstraintModel | None:
        """Get the constraint model.

        Returns
        -------
        ConstraintModel
        """
        return self._constraint

    @property
    def masks(self) -> dict[str, NDArray[np.bool_]]:
        """Get the masks for the parameters.

        Returns
        -------
        dict
        """
        return self._masks

    @property
    def continuous_dimensions(self) -> NDArray[np.bool_]:
        """Get the continuous parameters.

        Returns
        -------
        dict
        """
        result = np.zeros(self.dim, dtype=bool)
        masks = self.masks
        for key in self.keys:
            result[masks[key]] = self._params_config[key].is_continuous
        return result

    def make_params(self, pbounds: BoundsMapping) -> dict[str, BayesParameter]:
        """Create a dictionary of parameters from a dictionary of bounds.

        Parameters
        ----------
        pbounds : dict
            A dictionary with the parameter names as keys and a tuple with minimum
            and maximum values.

        Returns
        -------
        dict
            A dictionary with the parameter names as keys and the corresponding
            parameter objects as values.
        """
        any_is_not_float = False  # TODO: remove in an upcoming release
        params: dict[str, BayesParameter] = {}
        for key in pbounds:
            pbound = pbounds[key]

            if isinstance(pbound, BayesParameter):
                res = pbound
                if not isinstance(pbound, FloatParameter):
                    any_is_not_float = True
            elif (len(pbound) == 2 and is_numeric(pbound[0]) and is_numeric(pbound[1])) or (
                len(pbound) == 3 and pbound[-1] is float
            ):
                res = FloatParameter(name=key, bounds=(float(pbound[0]), float(pbound[1])))
            elif len(pbound) == 3 and pbound[-1] is int:
                res = IntParameter(name=key, bounds=(int(pbound[0]), int(pbound[1])))
                any_is_not_float = True
            else:
                # assume categorical variable with pbound as list of possible values
                res = CategoricalParameter(name=key, categories=pbound)
                any_is_not_float = True
            params[key] = res
        if any_is_not_float:
            msg = (
                "Non-float parameters are experimental and may not work as expected."
                " Exercise caution when using them and please report any issues you encounter."
            )
            warn(msg, stacklevel=4)
        return params

    def make_masks(self) -> dict[str, NDArray[np.bool_]]:
        """Create a dictionary of masks for the parameters.

        The mask can be used to select the corresponding parameters from an array.

        Returns
        -------
        dict
            A dictionary with the parameter names as keys and the corresponding
            mask as values.
        """
        masks = {}
        pos = 0
        for key in self._keys:
            mask = np.zeros(self._dim)
            mask[pos : pos + self._params_config[key].dim] = 1
            masks[key] = mask.astype(bool)
            pos = pos + self._params_config[key].dim
        return masks

    def calculate_bounds(self) -> NDArray[Float]:
        """Calculate the float bounds of the parameter space."""
        bounds = np.empty((self._dim, 2))
        for key in self._keys:
            bounds[self.masks[key]] = self._params_config[key].bounds
        return bounds

    def params_to_array(self, params: Mapping[str, float | NDArray[Float]]) -> NDArray[Float]:
        """Convert a dict representation of parameters into an array version.

        Parameters
        ----------
        params : dict
            a single point, with len(x) == self.dim.

        Returns
        -------
        np.ndarray
            Representation of the parameters as an array.
        """
        if set(params) != set(self.keys):
            error_msg = (
                f"Parameters' keys ({params}) do " f"not match the expected set of keys ({self.keys})."
            )
            raise ValueError(error_msg)
        return self._to_float(params)

    @property
    def constraint_values(self) -> NDArray[Float]:
        """Get the constraint values registered to this TargetSpace.

        Returns
        -------
        np.ndarray
        """
        if self._constraint is None:
            error_msg = "TargetSpace belongs to an unconstrained optimization"
            raise AttributeError(error_msg)

        return self._constraint_values

    def kernel_transform(self, value: NDArray[Float]) -> NDArray[Float]:
        """Transform floating-point suggestions to values used in the kernel.

        Vectorized.
        """
        value = np.atleast_2d(value)
        res = [self._params_config[p].kernel_transform(value[:, self.masks[p]]) for p in self._keys]
        return np.hstack(res)

    def array_to_params(self, x: NDArray[Float]) -> dict[str, float | NDArray[Float]]:
        """Convert an array representation of parameters into a dict version.

        Parameters
        ----------
        x : np.ndarray
            a single point, with len(x) == self.dim.

        Returns
        -------
        dict
            Representation of the parameters as dictionary.
        """
        if len(x) != self._dim:
            error_msg = (
                f"Size of array ({len(x)}) is different than the "
                f"expected number of parameters ({self._dim})."
            )
            raise ValueError(error_msg)
        return self._to_params(x)

    def _to_float(self, value: Mapping[str, float | NDArray[Float]]) -> NDArray[Float]:
        if set(value) != set(self.keys):
            msg = f"Parameters' keys ({value}) do " f"not match the expected set of keys ({self.keys})."
            raise ValueError(msg)
        res = np.zeros(self._dim)
        for key in self._keys:
            p = self._params_config[key]
            res[self.masks[key]] = p.to_float(value[key])
        return res

    def _to_params(self, value: NDArray[Float]) -> dict[str, float | NDArray[Float]]:
        res: dict[str, float | NDArray[Float]] = {}
        for key in self._keys:
            p = self._params_config[key]
            mask = self.masks[key]
            res[key] = p.to_param(value[mask])
        return res

    @property
    def mask(self) -> NDArray[np.bool_]:
        """Return a boolean array of valid points.

        Points are valid if they satisfy both the constraint and boundary conditions.

        Returns
        -------
        np.ndarray
        """
        mask = np.ones_like(self.target, dtype=bool)

        # mask points that don't satisfy the constraint
        if self._constraint is not None:
            mask &= self._constraint.allowed(self._constraint_values)

        # mask points that are outside the bounds
        if self._bounds is not None:
            within_bounds = np.all(
                (self._bounds[:, 0] <= self._params) & (self._params <= self._bounds[:, 1]), axis=1
            )
            mask &= within_bounds

        return mask

    def _as_array(self, x: Any) -> NDArray[Float]:
        try:
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)

        x = x.ravel()
        if x.size != self.dim:
            msg = f"Size of array ({len(x)}) is different than the expected number of ({self.dim})."
            raise ValueError(msg)
        return x

    def register(
        self, params: ParamsType, target: float, constraint_value: float | NDArray[Float] | None = None
    ) -> None:
        """Append a point and its target value to the known data.

        Parameters
        ----------
        params : np.ndarray
            a single point, with len(x) == self.dim.

        target : float
            target function value

        constraint_value : float or np.ndarray or None
            Constraint function value

        Raises
        ------
        NotUniqueError:
            if the point is not unique

        Notes
        -----
        runs in amortized constant time

        Examples
        --------
        >>> target_func = lambda p1, p2: p1 + p2
        >>> pbounds = {"p1": (0, 1), "p2": (1, 100)}
        >>> space = TargetSpace(target_func, pbounds)
        >>> len(space)
        0
        >>> x = np.array([0, 0])
        >>> y = 1
        >>> space.register(x, y)
        >>> len(space)
        1
        """
        x = self._as_array(params)

        if x in self:
            if self._allow_duplicate_points:
                self.n_duplicate_points = self.n_duplicate_points + 1

                print(
                    Fore.RED + f"Data point {x} is not unique. {self.n_duplicate_points}"
                    " duplicates registered. Continuing ..." + Fore.RESET
                )
            else:
                error_msg = (
                    f"Data point {x} is not unique. You can set"
                    ' "allow_duplicate_points=True" to avoid this error'
                )
                raise NotUniqueError(error_msg)

        # if x is not within the bounds of the parameter space, warn the user
        if self._bounds is not None and not np.all((self._bounds[:, 0] <= x) & (x <= self._bounds[:, 1])):
            for key in self.keys:
                if not np.all(
                    (self._params_config[key].bounds[..., 0] <= x[self.masks[key]])
                    & (x[self.masks[key]] <= self._params_config[key].bounds[..., 1])
                ):
                    msg = (
                        f"\nData point {x} is outside the bounds of the parameter {key}."
                        f"\n\tBounds:\n{self._params_config[key].bounds}"
                    )
                    warn(msg, stacklevel=2)

        # Make copies of the data, so as not to modify the originals incase something fails
        # during the registration process. This prevents out-of-sync data.
        params_copy: NDArray[Float] = np.concatenate([self._params, x.reshape(1, -1)])
        target_copy: NDArray[Float] = np.concatenate([self._target, [target]])
        cache_copy = self._cache.copy()  # shallow copy suffices

        if self._constraint is None:
            # Insert data into unique dictionary
            cache_copy[_hashable(x.ravel())] = target
        else:
            if constraint_value is None:
                msg = (
                    "When registering a point to a constrained TargetSpace"
                    " a constraint value needs to be present."
                )
                raise ValueError(msg)
            # Insert data into unique dictionary
            cache_copy[_hashable(x.ravel())] = (target, constraint_value)
            constraint_values_copy: NDArray[Float] = np.concatenate(
                [self._constraint_values, [constraint_value]]
            )
            self._constraint_values = constraint_values_copy

        # Operations passed, update the variables
        self._params = params_copy
        self._target = target_copy
        self._cache = cache_copy

    def probe(self, params: ParamsType) -> float | tuple[float, float | NDArray[Float]]:
        """Evaluate the target function on a point and register the result.

        Notes
        -----
        If `params` has been previously seen and duplicate points are not allowed,
        returns a cached value of `result`.

        Parameters
        ----------
        params : np.ndarray
            a single point, with len(x) == self.dim

        Returns
        -------
        result : float | Tuple(float, float)
            target function value, or Tuple(target function value, constraint value)

        Example
        -------
        >>> target_func = lambda p1, p2: p1 + p2
        >>> pbounds = {"p1": (0, 1), "p2": (1, 100)}
        >>> space = TargetSpace(target_func, pbounds)
        >>> space.probe([1, 5])
        >>> assert self.max()["target"] == 6
        >>> assert self.max()["params"] == {"p1": 1.0, "p2": 5.0}
        """
        x = self._as_array(params)
        if x in self and not self._allow_duplicate_points:
            return self._cache[_hashable(x.ravel())]

        dict_params = self.array_to_params(x)
        if self.target_func is None:
            error_msg = "No target function has been provided."
            raise ValueError(error_msg)
        target = self.target_func(**dict_params)

        if self._constraint is None:
            self.register(x, target)
            return target

        constraint_value = self._constraint.eval(**dict_params)
        self.register(x, target, constraint_value)
        return target, constraint_value

    def random_sample(
        self, n_samples: int = 0, random_state: np.random.RandomState | int | None = None
    ) -> NDArray[Float]:
        """
        Sample a random point from within the bounds of the space.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to draw. If 0, a single sample is drawn,
            and a 1D array is returned. If n_samples > 0, an array of
            shape (n_samples, dim) is returned.

        random_state : np.random.RandomState | int | None
            The random state to use for sampling.

        Returns
        -------
        data: ndarray
            [1 x dim] array with dimensions corresponding to `self._keys`

        Examples
        --------
        >>> target_func = lambda p1, p2: p1 + p2
        >>> pbounds = {"p1": (0, 1), "p2": (1, 100)}
        >>> space = TargetSpace(target_func, pbounds, random_state=0)
        >>> space.random_sample()
        array([[ 0.54488318,   55.33253689]])
        """
        random_state = ensure_rng(random_state)
        flatten = n_samples == 0
        n_samples = max(1, n_samples)
        data = np.empty((n_samples, self._dim))
        for key, mask in self.masks.items():
            smpl = self._params_config[key].random_sample(n_samples, random_state)
            data[:, mask] = smpl.reshape(n_samples, self._params_config[key].dim)
        if flatten:
            return data.ravel()
        return data

    def _target_max(self) -> float | None:
        """Get the maximum target value within the current parameter bounds.

        If there is a constraint present, the maximum value that fulfills the
        constraint within the parameter bounds is returned.

        Returns
        -------
        max: float
            The maximum target value.
        """
        if len(self.target) == 0:
            return None

        if len(self.target[self.mask]) == 0:
            return None

        return self.target[self.mask].max()

    def max(self) -> dict[str, Any] | None:
        """Get maximum target value found and corresponding parameters.

        If there is a constraint present, the maximum value that fulfills the
        constraint within the parameter bounds is returned.

        Returns
        -------
        res: dict
            A dictionary with the keys 'target' and 'params'. The value of
            'target' is the maximum target value, and the value of 'params' is
            a dictionary with the parameter names as keys and the parameter
            values as values.
        """
        target_max = self._target_max()
        if target_max is None:
            return None

        target = self.target[self.mask]
        params = self.params[self.mask]
        target_max_idx = np.argmax(target)

        res = {"target": target_max, "params": dict(zip(self.keys, params[target_max_idx]))}

        if self._constraint is not None:
            constraint_values = self.constraint_values[self.mask]
            res["constraint"] = constraint_values[target_max_idx]

        return res

    def res(self) -> list[dict[str, Any]]:
        """Get all target values and constraint fulfillment for all parameters.

        Returns
        -------
        res: list
            A list of dictionaries with the keys 'target', 'params', and
            'constraint'. The value of 'target' is the target value, the value
            of 'params' is a dictionary with the parameter names as keys and the
            parameter values as values, and the value of 'constraint' is the
            constraint fulfillment.

        Notes
        -----
        Does not report if points are within the bounds of the parameter space.
        """
        if self._constraint is None:
            params = [self.array_to_params(p) for p in self.params]

            return [{"target": target, "params": param} for target, param in zip(self.target, params)]

        params = [dict(zip(self.keys, p)) for p in self.params]

        return [
            {"target": target, "constraint": constraint_value, "params": param, "allowed": allowed}
            for target, constraint_value, param, allowed in zip(
                self.target,
                self._constraint_values,
                params,
                self._constraint.allowed(self._constraint_values),
            )
        ]

    def set_bounds(self, new_bounds: BoundsMapping) -> None:
        """Change the lower and upper search bounds.

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        new_params_config = self.make_params(new_bounds)

        dims = 0
        params_config = deepcopy(self._params_config)
        for key in self.keys:
            if key in new_bounds:
                if not isinstance(new_params_config[key], type(self._params_config[key])):
                    msg = (
                        f"Parameter type {type(new_params_config[key])} of"
                        " new bounds does not match parameter type"
                        f" {type(self._params_config[key])} of old bounds"
                    )
                    raise ValueError(msg)
                params_config[key] = new_params_config[key]
            dims = dims + params_config[key].dim
        if dims != self.dim:
            msg = (
                f"Dimensions of new bounds ({dims}) does not match" f" dimensions of old bounds ({self.dim})."
            )
            raise ValueError(msg)
        self._params_config = params_config
        self._bounds = self.calculate_bounds()
