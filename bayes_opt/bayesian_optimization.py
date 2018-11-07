import warnings
import numpy as np

from .target_space import TargetSpace
from .observer import Observable, Events
from .helpers import UtilityFunction, acq_max, ensure_rng

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor


class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise ValueError("Cannot retrieve next object from empty queue.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class BayesianOptimization(Observable):
    def __init__(self, f, pbounds, random_state=None, verbose=1):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-5,
            normalize_y=True,
            n_restarts_optimizer=25,
            random_state=self._random_state,
        )

        super(BayesianOptimization, self).__init__(events=None)

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, x, target):
        """Expect observation with known target"""
        self._space.register(x, target)

    def probe(self, x, lazy=True):
        """Probe target of x"""
        if lazy:
            self._queue.add(x)
        else:
            self._space.probe(x)

    def suggest(self, utility_function):
        """Most promissing point to probe next"""
        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.x, self._space.target)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )

        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            self._queue.add(self._space.random_sample())

    def maximize(self,
                 init_points: int=5,
                 n_iter: int=25,
                 acq: str='ucb',
                 kappa: float=2.576,
                 xi: float=0.0,
                 **gp_params):
        """Mazimize your function"""
        self._prime_queue(init_points)

        util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except ValueError:
                x_probe = self.suggest(util)
                iteration += 1

            self.probe(x_probe, lazy=False)

        # Notify about finished optimization
        self.dispatch(Events.FIT_DONE)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)
