from __future__ import print_function, division
import numpy as np
from .helpers import ensure_rng, unique_rows


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
    >>> y = space.observe_point(x)
    >>> assert self.max_point()['max_val'] == y
    """
    def __init__(self, target_func, pbounds, random_state=None):
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
        """

        self.random_state = ensure_rng(random_state)

        # Some function to be optimized
        self.target_func = target_func

        # Get the name of the parameters
        self.keys = list(pbounds.keys())
        # Create an array with parameters bounds
        self.bounds = np.array(list(pbounds.values()), dtype=np.float)
        # Find number of parameters
        self.dim = len(self.keys)

        # preallocated memory for X and Y points
        self._Xarr = None
        self._Yarr = None

        # Number of observations
        self._length = 0

        # Views of the preallocated arrays showing only populated data
        self._Xview = None
        self._Yview = None

        self._cache = {}  # keep track of unique points we have seen so far

    @property
    def X(self):
        return self._Xview

    @property
    def Y(self):
        return self._Yview

    def __contains__(self, x):
        return _hashable(x) in self._cache

    def __len__(self):
        return self._length

    def _dict_to_points(self, points_dict):
        """
        Example:
        -------
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(lambda p1, p2: p1 + p2, pbounds)
        >>> points_dict = {'p1': [0, .5, 1], 'p2': [0, 1, 2]}
        >>> space._dict_to_points(points_dict)
        [[0, 0], [1, 0.5], [2, 1]]
        """
        # Consistency check
        param_tup_lens = []

        for key in self.keys:
            param_tup_lens.append(len(list(points_dict[key])))

        if all([e == param_tup_lens[0] for e in param_tup_lens]):
            pass
        else:
            raise ValueError('The same number of initialization points '
                             'must be entered for every parameter.')

        # Turn into list of lists
        all_points = []
        for key in self.keys:
            all_points.append(points_dict[key])

        # Take transpose of list
        points = list(map(list, zip(*all_points)))
        return points

    def observe_point(self, x):
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
        x = np.asarray(x).ravel()
        assert x.size == self.dim, 'x must have the same dimensions'

        if x in self:
            # Lookup previously seen point
            y = self._cache[_hashable(x)]
        else:
            # measure the target function
            params = dict(zip(self.keys, x))
            y = self.target_func(**params)
            self.add_observation(x, y)
        return y

    def add_observation(self, x, y):
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
        if x in self:
            raise KeyError('Data point {} is not unique'.format(x))

        if self._length >= self._n_alloc_rows:
            self._allocate((self._length + 1) * 2)

        x = np.asarray(x).ravel()

        # Insert data into unique dictionary
        self._cache[_hashable(x)] = y

        # Insert data into preallocated arrays
        self._Xarr[self._length] = x
        self._Yarr[self._length] = y
        # Expand views to encompass the new data point
        self._length += 1

        # Create views of the data
        self._Xview = self._Xarr[:self._length]
        self._Yview = self._Yarr[:self._length]

    def _allocate(self, num):
        """
        Allocate enough memory to store `num` points
        """
        if num <= self._n_alloc_rows:
            raise ValueError('num must be larger than current array length')

        self._assert_internal_invariants()

        # Allocate new memory
        _Xnew = np.empty((num, self.bounds.shape[0]))
        _Ynew = np.empty(num)

        # Copy the old data into the new
        if self._Xarr is not None:
            _Xnew[:self._length] = self._Xarr[:self._length]
            _Ynew[:self._length] = self._Yarr[:self._length]
        self._Xarr = _Xnew
        self._Yarr = _Ynew

        # Create views of the data
        self._Xview = self._Xarr[:self._length]
        self._Yview = self._Yarr[:self._length]

    @property
    def _n_alloc_rows(self):
        """ Number of allocated rows """
        return 0 if self._Xarr is None else self._Xarr.shape[0]

    def random_points(self, num):
        """
        Creates random points within the bounds of the space

        Parameters
        ----------
        num : int
            Number of random points to create

        Returns
        ----------
        data: ndarray
            [num x dim] array points with dimensions corresponding to `self.keys`

        Example
        -------
        >>> target_func = lambda p1, p2: p1 + p2
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(target_func, pbounds, random_state=0)
        >>> space.random_points(3)
        array([[ 55.33253689,   0.54488318],
               [ 71.80374727,   0.4236548 ],
               [ 60.67357423,   0.64589411]])
        """
        # TODO: support integer, category, and basic scipy.optimize constraints
        data = np.empty((num, self.dim))
        for col, (lower, upper) in enumerate(self.bounds):
            data.T[col] = self.random_state.uniform(lower, upper, size=num)
        return data

    def max_point(self):
        """
        Return the current parameters that best maximize target function with
        that maximum value.
        """
        return {'max_val': self.Y.max(),
                'max_params': dict(zip(self.keys,
                                       self.X[self.Y.argmax()]))}

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        # Loop through the all bounds and reset the min-max bound matrix
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                self.bounds[row] = new_bounds[key]

    def _assert_internal_invariants(self, fast=True):
        """
        Run internal consistency checks to ensure that data structure
        assumptions have not been violated.
        """
        if self._Xarr is None:
            assert self._Yarr is None
            assert self._Xview is None
            assert self._Yview is None
        else:
            assert self._Yarr is not None
            assert self._Xview is not None
            assert self._Yview is not None
            assert len(self._Xview) == self._length
            assert len(self._Yview) == self._length
            assert len(self._Xarr) == len(self._Yarr)

            if not fast:
                # run slower checks
                assert np.all(unique_rows(self.X))
                # assert np.may_share_memory(self._Xview, self._Xarr)
                # assert np.may_share_memory(self._Yview, self._Yarr)
