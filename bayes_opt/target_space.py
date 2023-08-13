"""Manages the optimization domain and holds points."""
import numpy as np
from colorama import Fore
from .util import ensure_rng, NotUniqueError


def _hashable(x):
    """Ensure that a point is hashable by a python dict."""
    return tuple(map(float, x))


class TargetSpace():
    """Holds the param-space coordinates (X) and target values (Y).

    Allows for constant-time appends.

    Parameters
    ----------
    target_func : function
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
    >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    >>> space = TargetSpace(target_func, pbounds, random_state=0)
    >>> x = space.random_points(1)[0]
    >>> y = space.register_point(x)
    >>> assert self.max_point()['max_val'] == y
    """

    def __init__(self, target_func, pbounds, constraint=None, random_state=None,
                 allow_duplicate_points=False):
        self.random_state = ensure_rng(random_state)
        self._allow_duplicate_points = allow_duplicate_points
        self.n_duplicate_points = 0

        # The function to be optimized
        self.target_func = target_func

        # Get the name of the parameters
        self._keys = sorted(pbounds)
        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],
            dtype=float
        )

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0,))

        # keep track of unique points we have seen so far
        self._cache = {}

        self._constraint = constraint

        if constraint is not None:
            # preallocated memory for constraint fulfillment
            if constraint.lb.size == 1:
                self._constraint_values = np.empty(shape=(0), dtype=float)
            else:
                self._constraint_values = np.empty(shape=(0, constraint.lb.size), dtype=float)

    def __contains__(self, x):
        """Check if this parameter has already been registered.
        
        Returns
        -------
        bool
        """
        return _hashable(x) in self._cache

    def __len__(self):
        """Return number of observations registered.
        
        Returns
        -------
        int
        """
        assert len(self._params) == len(self._target)
        return len(self._target)

    @property
    def empty(self):
        """Check if anything has been registered.
        
        Returns
        -------
        bool
        """
        return len(self) == 0

    @property
    def params(self):
        """Get the parameter values registered to this TargetSpace.
        
        Returns
        -------
        np.ndarray
        """
        return self._params

    @property
    def target(self):
        """Get the target function values registered to this TargetSpace.
        
        Returns
        -------
        np.ndarray
        """
        return self._target

    @property
    def dim(self):
        """Get the number of parameter names.
        
        Returns
        -------
        int
        """
        return len(self._keys)

    @property
    def keys(self):
        """Get the keys (or parameter names).
        
        Returns
        -------
        list of str
        """
        return self._keys

    @property
    def bounds(self):
        """Get the bounds of this TargetSpace.
        
        Returns
        -------
        np.ndarray
        """
        return self._bounds

    @property
    def constraint(self):
        """Get the constraint model.
        
        Returns
        -------
        ConstraintModel
        """
        return self._constraint

    @property
    def constraint_values(self):
        """Get the constraint values registered to this TargetSpace.
        
        Returns
        -------
        np.ndarray
        """
        if self._constraint is None:
            raise AttributeError("TargetSpace belongs to an unconstrained optimization")

        return self._constraint_values

    def params_to_array(self, params):
        """Convert a dict representation of parameters into an array version.

        Parameters
        ----------
        params : dict
            a single point, with len(x) == self.dim.

        Returns
        -------
        np.ndarray
            Representation of the parameters as dictionary.
        """
        try:
            assert set(params) == set(self.keys)
        except AssertionError as e:
            raise ValueError(
                f"Parameters' keys ({sorted(params)}) do " +
                f"not match the expected set of keys ({self.keys})."
            ) from e
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
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
        try:
            assert len(x) == len(self.keys)
        except AssertionError as e:
            raise ValueError(
                f"Size of array ({len(x)}) is different than the " +
                f"expected number of parameters ({len(self.keys)})."
            ) from e
        return dict(zip(self.keys, x))

    def _as_array(self, x):
        try:
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)

        x = x.ravel()
        try:
            assert x.size == self.dim
        except AssertionError as e:
            raise ValueError(
                f"Size of array ({len(x)}) is different than the " +
                f"expected number of parameters ({len(self.keys)})."
            ) from e
        return x

    def register(self, params, target, constraint_value=None):
        """Append a point and its target value to the known data.

        Parameters
        ----------
        x : np.ndarray
            a single point, with len(x) == self.dim.

        y : float
            target function value

        Raises
        ------
        KeyError:
            if the point is not unique

        Notes
        -----
        runs in amortized constant time

        Examples
        --------
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(lambda p1, p2: p1 + p2, pbounds)
        >>> len(space)
        0
        >>> x = np.array([0, 0])
        >>> y = 1
        >>> space.add_observation(x, y)
        >>> len(space)
        1
        """
        x = self._as_array(params)
        if x in self:
            if self._allow_duplicate_points:
                self.n_duplicate_points = self.n_duplicate_points + 1

                print(Fore.RED + f'Data point {x} is not unique. {self.n_duplicate_points}'
                              ' duplicates registered. Continuing ...')
            else:
                raise NotUniqueError(f'Data point {x} is not unique. You can set'
                                     ' "allow_duplicate_points=True" to avoid this error')

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

        if self._constraint is None:
            # Insert data into unique dictionary
            self._cache[_hashable(x.ravel())] = target
        else:
            if constraint_value is None:
                msg = ("When registering a point to a constrained TargetSpace" +
                       " a constraint value needs to be present.")
                raise ValueError(msg)
            # Insert data into unique dictionary
            self._cache[_hashable(x.ravel())] = (target, constraint_value)
            self._constraint_values = np.concatenate([self._constraint_values,
                                                      [constraint_value]])

    def probe(self, params):
        """Evaluate the target function on a point and register the result.

        Notes
        -----
        If `params` has been previously seen returns a cached value of `y`.

        Parameters
        ----------
        params : np.ndarray
            a single point, with len(x) == self.dim

        Returns
        -------
        y : float
            target function value.
        """
        x = self._as_array(params)
        params = dict(zip(self._keys, x))
        target = self.target_func(**params)

        if self._constraint is None:
            self.register(x, target)
            return target

        constraint_value = self._constraint.eval(**params)
        self.register(x, target, constraint_value)
        return target, constraint_value

    def random_sample(self):
        """Create random points within the bounds of the space.

        Returns
        -------
        data: np.ndarray
            [num x dim] array points with dimensions corresponding to `self._keys`

        Examples
        --------
        >>> target_func = lambda p1, p2: p1 + p2
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(target_func, pbounds, random_state=0)
        >>> space.random_points(1)
        array([[ 55.33253689,   0.54488318]])
        """
        data = np.empty((1, self.dim))
        for col, (lower, upper) in enumerate(self._bounds):
            data.T[col] = self.random_state.uniform(lower, upper, size=1)
        return data.ravel()

    def _target_max(self):
        """Get maximum target value found.
        
        If there is a constraint present, the maximum value that fulfills the
        constraint is returned.

        
        Returns
        -------
        dict | None
            The maximum allowed point's target function value.
            Returns None if there is no (allowed) maximum.
        """
        if len(self.target) == 0:
            return None

        if self._constraint is None:
            return self.target.max()

        allowed = self._constraint.allowed(self._constraint_values)
        if allowed.any():
            return self.target[allowed].max()

        return None

    def max(self):
        """Get maximum target value found and corresponding parameters.
        
        If there is a constraint present, the maximum value that fulfills the
        constraint is returned.

        Returns
        -------
        dict | None
            A dictionary containing the maximum allowed point's target function
            value, parameters, and, if applicable, constraint function value.
            Returns None if there is no (allowed) maximum.
        """
        target_max = self._target_max()

        if target_max is None:
            return None

        if self._constraint is not None:
            allowed = self._constraint.allowed(self._constraint_values)

            target = self.target[allowed]
            params = self.params[allowed]
            constraint_values = self.constraint_values[allowed]
        else:
            target = self.target
            params = self.params

        target_max_idx = np.where(target == target_max)[0][0]


        res = {
                'target': target_max,
                'params': dict(
                zip(self.keys, params[target_max_idx])
            )
        }

        if self._constraint is not None:
            res['constraint'] = constraint_values[target_max_idx]

        return res

    def res(self):
        """Get all target values and constraint fulfillment for all parameters.

        Returns
        -------
        dict
            A dictionary containing the target function values, parameters,
            and, if applicable, constraint function values and allowedness.
        """
        if self._constraint is None:
            params = [dict(zip(self.keys, p)) for p in self.params]

            return [
                {"target": target, "params": param}
                for target, param in zip(self.target, params)
            ]

        params = [dict(zip(self.keys, p)) for p in self.params]

        return [
            {
                "target": target,
                "constraint": constraint_value,
                "params": param,
                "allowed": allowed
            }
            for target, constraint_value, param, allowed in zip(
                self.target,
                self._constraint_values,
                params,
                self._constraint.allowed(self._constraint_values)
            )
        ]

    def set_bounds(self, new_bounds):
        """Change the lower and upper search bounds.

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                self._bounds[row] = new_bounds[key]
