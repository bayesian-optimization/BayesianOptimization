import numpy as np
from .util import ensure_rng, NotUniqueError
from .parameter import FloatParameter, IntParameter, is_numeric, CategoricalParameter
from .constraint import ConstraintModel


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

    def __init__(self,
                 target_func,
                 pbounds,
                 constraint=None,
                 random_state=None):
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

        # The function to be optimized
        self.target_func = target_func

        self._keys = sorted(pbounds)
        self._params_config = self.make_params(pbounds)
        self._dim = sum([self._params_config[key].dim for key in self._keys])

        self._masks = self.make_masks()
        self._float_bounds = self.calculate_float_bounds()

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0))

        # keep track of unique points we have seen so far
        self._cache = {}

        if constraint is not None:
            self._constraint = ConstraintModel(constraint.fun,
                                               constraint.lb,
                                               constraint.ub,
                                               transform=self.kernel_transform,
                                               random_state=random_state)
            # preallocated memory for constraint fulfillement
            if self._constraint.lb.size == 1:
                self._constraint_values = np.empty(shape=(0), dtype=float)
            else:
                self._constraint_values = np.empty(
                    shape=(0, self._constraint.lb.size), dtype=float)
        else:
            self._constraint = None

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
        return self._dim

    @property
    def bounds(self):
        return [self._params_config[key].domain for key in self.keys]

    @property
    def constraint(self):
        return self._constraint

    @property
    def constraint_values(self):
        if self._constraint is not None:
            return self._constraint_values

    @property
    def keys(self):
        return self._keys

    @property
    def float_bounds(self):
        return self._float_bounds

    @property
    def masks(self):
        return self._masks

    def make_params(self, pbounds) -> dict:
        params = {}
        for key in sorted(pbounds):
            pbound = pbounds[key]
            if len(pbound) == 2 and is_numeric(pbound[0]) and is_numeric(
                    pbound[1]):
                res = FloatParameter(name=key, domain=pbound)
            elif len(pbound) == 3 and pbound[-1] == float:
                res = FloatParameter(name=key, domain=(pbound[0], pbound[1]))
            elif len(pbound) == 3 and pbound[-1] == int:
                res = IntParameter(name=key, domain=(int(pbound[0]), int(pbound[1])))
            else:
                # assume categorical variable with pbound as list of possible values
                res = CategoricalParameter(name=key, domain=pbound)
            params[key] = res
        return params

    def make_masks(self):
        masks = {}
        pos = 0
        for key in self._keys:
            mask = np.zeros(self._dim)
            mask[pos:pos + self._params_config[key].dim] = 1
            masks[key] = mask.astype(bool)
            pos = pos + self._params_config[key].dim
        return masks

    def calculate_float_bounds(self):
        bounds = np.empty((self._dim, 2))
        for key in self._keys:
            bounds[self.masks[key]] = self._params_config[key].float_bounds
        return bounds

    def params_to_array(self, value) -> np.ndarray:
        if type(value) == dict:
            # assume the input is one single set of parameters
            return self._to_float(value)
        else:
            return np.vstack([self._to_float(x) for x in value])

    def _to_float(self, value) -> np.ndarray:
        try:
            assert set(value) == set(self.keys)
        except AssertionError:
            raise ValueError(
                f"Parameters' keys ({sorted(value)}) do " +
                f"not match the expected set of keys ({self.keys}).")
        res = np.zeros(self._dim)
        for key in self._keys:
            p = self._params_config[key]
            res[self._masks[key]] = p.to_float(value[key])
        return res

    def array_to_params(self, value: np.ndarray):
        try:
            assert value.shape[-1] == self._dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(
                    value.shape[-1]) +
                "expected number of parameters ({}).".format(self._dim))
        if len(value.shape) == 1:
            return self._to_params(value)
        else:
            return [self._to_params(v) for v in value]

    def _to_params(self, value: np.ndarray) -> dict:
        res = {}
        for key in self._keys:
            p = self._params_config[key]
            mask = self._masks[key]
            res[key] = p.to_param(value[mask])
        return res

    def kernel_transform(self, value: np.ndarray) -> np.ndarray:
        """Transform floating-point suggestions to values used in the kernel.
        
        Vectorized."""
        value = np.atleast_2d(value)
        res = []
        for p in self._keys:
            par = self._params_config[p].kernel_transform(value[:,
                                                                self.masks[p]])
            res.append(par)
        return np.hstack(res)

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

        if type(params) == np.ndarray:
            x = params
        else:
            assert type(params) == dict
            x = self.params_to_array(params)

        if x in self:
            raise NotUniqueError('Data point {} is not unique'.format(x))

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, np.atleast_1d(target)])

        if self._constraint is None:
            # Insert data into unique dictionary
            self._cache[_hashable(x.ravel())] = target
        else:
            if constraint_value is None:
                msg = (
                    "When registering a point to a constrained TargetSpace" +
                    " a constraint value needs to be present.")
                raise ValueError(msg)
            # Insert data into unique dictionary
            self._cache[_hashable(x.ravel())] = (target, constraint_value)
            self._constraint_values = np.concatenate(
                [self._constraint_values,
                 np.atleast_1d(constraint_value)])

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
        if type(params) == np.ndarray:
            x = params
            params = self.array_to_params(params)
        else:
            assert type(params) == dict
            x = self.params_to_array(params)

        try:
            return self._cache[_hashable(x)]
        except KeyError:
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
            [num x dim] array points with dimensions corresponding to `self.keys`

        Example
        -------
        >>> target_func = lambda p1, p2: p1 + p2
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(target_func, pbounds, random_state=0)
        >>> space.random_points(1)
        array([[ 55.33253689,   0.54488318]])
        """
        data = np.empty((1, self._dim))
        for col, (lower, upper) in enumerate(self._float_bounds):
            data.T[col] = self.random_state.uniform(lower, upper, size=1)
        return data.ravel()

    def max(self):
        """Get maximum target value found and corresponding parameters.
        
        If there is a constraint present, the maximum value that fulfills the
        constraint is returned."""
        if self._constraint is None:
            try:
                res = {
                    'target':
                    self.target.max(),
                    'params':
                    self.array_to_params(self.params[self.target.argmax()])
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
                    'params': dict(zip(self.keys, self.params[idx])),
                    'constraint': self._constraint_values[idx]
                }
            else:
                res = {'target': None, 'params': None, 'constraint': None}
            return res

    def res(self):
        """Get all target values and constraint fulfillment for all parameters.
        """
        if self._constraint is None:
            params = [self.array_to_params(p) for p in self.params]

            return [{
                "target": target,
                "params": param
            } for target, param in zip(self.target, params)]
        else:
            params = params = [self.array_to_params(p) for p in self.params]
            return [{
                "target": target,
                "constraint": constraint_value,
                "params": param,
                "allowed": allowed
            } for target, constraint_value, param, allowed in zip(
                self.target, self._constraint_values, params,
                self._constraint.allowed(self._constraint_values))]

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        new__params_config = self.make_params(new_bounds)


        for row, key in enumerate(self.keys):
            if key in new_bounds:
                if isinstance(self._params_config[key], CategoricalParameter):
                    if set(self._params_config[key].domain) == set(new_bounds[key]):
                        msg = "Changing bounds of categorical parameters is not supported"
                        raise NotImplementedError(msg)
                self._params_config[key] = new__params_config[key]
        
        self._dim = sum([self._params_config[key].dim for key in self._keys])
        self._masks = self.make_masks()
        self._float_bounds = self.calculate_float_bounds()
