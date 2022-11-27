import warnings

import numpy as np
from .util import ensure_rng, NotUniqueError
from .util import Colours


def _hashable(x):
    """ ensure that an point is hashable by a python dict """
    return tuple(map(float, x))


class TargetSpace(object):
    """
    Holds the param-space coordinates (X) and target values (Y)
    Allows for constant-time appends while ensuring no duplicates are added

    Example
    -------
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
        """
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
            may occasionaly generate a duplicate point.
        """
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
        self._target = np.empty(shape=(0))

        # keep track of unique points we have seen so far
        self._cache = {}

        self._constraint = constraint

        if constraint is not None:
            # preallocated memory for constraint fulfillement
            if constraint.lb.size == 1:
                self._constraint_values = np.empty(shape=(0), dtype=float)
            else:
                self._constraint_values = np.empty(shape=(0, constraint.lb.size), dtype=float)

    def __contains__(self, x):
        return _hashable(x) in self._cache

    def __len__(self):
        assert len(self._params) == len(self._target)
        return len(self._target)

    @property
    def empty(self):
        return len(self) == 0

    @property
    def params(self):
        return self._params

    @property
    def target(self):
        return self._target

    @property
    def dim(self):
        return len(self._keys)

    @property
    def keys(self):
        return self._keys

    @property
    def bounds(self):
        return self._bounds

    @property
    def constraint(self):
        return self._constraint

    @property
    def constraint_values(self):
        if self._constraint is not None:
            return self._constraint_values

    def params_to_array(self, params):
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return dict(zip(self.keys, x))

    def _as_array(self, x):
        try:
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)

        x = x.ravel()
        try:
            assert x.size == self.dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys)))
        return x

    def register(self, params, target, constraint_value=None):
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        y : float
            target function value

        Raises
        ------
        KeyError:
            if the point is not unique

        Notes
        -----
        runs in ammortized constant time

        Example
        -------
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
                print(f'{Colours.RED}Data point {x} is not unique. {self.n_duplicate_points} duplicates registered.'
                              f' Continuing ...{Colours.END}')
            else:
                raise NotUniqueError(f'Data point {x} is not unique. You can set "allow_duplicate_points=True" to '
                                     f'avoid this error')

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
        """
        Evaulates a single point x, to obtain the value y and then records them
        as observations.

        Notes
        -----
        If x has been previously seen returns a cached value of y.

        Parameters
        ----------
        x : ndarray
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
        else:
            constraint_value = self._constraint.eval(**params)
            self.register(x, target, constraint_value)
            return target, constraint_value

    def random_sample(self):
        """
        Creates random points within the bounds of the space.

        Returns
        ----------
        data: ndarray
            [num x dim] array points with dimensions corresponding to `self._keys`

        Example
        -------
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

    def max(self):
        """Get maximum target value found and corresponding parameters.
        
        If there is a constraint present, the maximum value that fulfills the
        constraint is returned."""
        if self._constraint is None:
            try:
                res = {
                    'target': self.target.max(),
                    'params': dict(
                        zip(self.keys, self.params[self.target.argmax()])
                    )
                }
            except ValueError:
                res = {}
            return res
        else:
            allowed = self._constraint.allowed(self._constraint_values)
            if allowed.any():
                # Getting of all points that fulfill the constraints, find the
                # one with the maximum value for the target function.
                sorted = np.argsort(self.target)
                idx = sorted[allowed[sorted]][-1]
                # there must be a better way to do this, right?
                res = {
                    'target': self.target[idx],
                    'params': dict(
                        zip(self.keys, self.params[idx])
                    ),
                    'constraint': self._constraint_values[idx]
                }
            else:
                res = {
                    'target': None,
                    'params': None,
                    'constraint': None
                }
            return res

    def res(self):
        """Get all target values and constraint fulfillment for all parameters.
        """
        if self._constraint is None:
            params = [dict(zip(self.keys, p)) for p in self.params]

            return [
                {"target": target, "params": param}
                for target, param in zip(self.target, params)
            ]
        else:
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
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                self._bounds[row] = new_bounds[key]
