import numpy as np
from .util import ensure_rng


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
    >>> ptypes = {'p1': float, 'p2': int}
    >>> space = TargetSpace(target_func, pbounds, ptypes, random_state=0)
    >>> x = space.random_points(1)[0]
    >>> y = space.register_point(x)
    >>> assert self.max_point()['max_val'] == y
    """
    def __init__(self, target_func, pbounds, ptypes=None,random_state=None):
        """
        Parameters
        ----------
        target_func : function
            Function to be maximized.

        pbounds : dict
            Dictionary with parameter names and list of the minimum and maximum boundaries
            and maximum values.

        ptypes : dict
            Dictionnary with parameter names and their type

        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator
        """
        self.random_state = ensure_rng(random_state)

        # The function to be optimized
        self.target_func = target_func

        # Get the name of the parameters
        self._keys = sorted(pbounds)
        # Create an array with parameters bounds
        self._bounds = np.array([list(pbounds[item]) for item in self._keys], dtype=float)
        # Create an array with the parameters type if declared
        if ptypes is None:
            self._btypes = None
        else:
            ## TODO: add exception if parameter names in btypes and ptypes do not have the same length and content
            ## TODO: or store pbounds and ptypes has dictionnaries
            try:
                assert (len(ptypes) == len(pbounds))
            except AssertionError:
                raise AssertionError("ptypes and pbounds do not have same content."+\
                                     "ptypes and pbounds must list exact same parameters")
            self._btypes = np.array([ptypes[item] for item in self._keys], dtype=type)

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0))

        # keep track of unique points we have seen so far
        self._cache = {}

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
    def btypes(self):
        return self._btypes

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
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return x

    def register(self, params, target):
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
        runs in amortized constant time

        Example
        -------
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> ptypes = {'p1': float, 'p2':int}
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
            raise KeyError('Data point {} is not unique'.format(x))

        # Insert data into unique dictionary
        self._cache[_hashable(x.ravel())] = target

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

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

        try:
            target = self._cache[_hashable(x)]
        except KeyError:
            params = dict(zip(self._keys, x))
            target = self.target_func(**params)
            self.register(x, target)
        return target

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
        >>> ptypes = {'p1': float, 'p2':int}
        >>> space = TargetSpace(target_func, pbounds, random_state=0)
        >>> space.random_points(1)
        array([[ 0.54488318, 55]])
        """
        # TODO: support category, and basic scipy.optimize constraints
        data = np.empty((1, self.dim))
        if self.btypes is None:
            for col, (lower, upper) in enumerate(self._bounds):
                data.T[col] = self.random_state.uniform(lower, upper, size=1)
        else:
            for col, (lower, upper) in enumerate(self._bounds):
                if self.btypes[col] != int:
                    data.T[col] = self.random_state.uniform(lower, upper, size=1)
                if self.btypes[col] == int:
                    data.T[col] = self.random_state.randint(int(lower), int(upper), size=1)
        return data.ravel()

    def max(self):
        """Get maximum target value found and corresponding parameters."""
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

    def res(self):
        """Get all target values found and corresponding parametes."""
        params = [dict(zip(self.keys, p)) for p in self.params]

        return [
            {"target": target, "params": param}
            for target, param in zip(self.target, params)
        ]

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds

        Returns
        ----------
        if type of modified parameter is int, then return rounded integer value
        Example : new_bounds = {"p1", (1.2, 8.7)} and "p1" is integer
        then new_bounds are (1,9)
        """
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                if self._btypes is not None:
                    if self._btypes[row] == int:
                        lbound = self._btypes[row](np.round(new_bounds[key][0], 0))
                        ubound = self._btypes[row](np.round(new_bounds[key][1], 0))
                        new_bounds[key] = (lbound, ubound)
                    self._bounds[row] = list(new_bounds[key])
                self._bounds[row] = list(new_bounds[key])
