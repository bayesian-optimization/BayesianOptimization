"""Main module.

Holds the `BayesianOptimization` class, which handles the maximization of a
function over a specific target space.
"""
from bayes_opt.constraint import ConstraintModel

from .target_space import TargetSpace
from .event import Events, DEFAULT_EVENTS
from .logger import _get_default_logger
from .util import ensure_rng

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from . import acquisition

class Queue:
    """Queue datastructure.

    Append items in the end, remove items from the front.
    """

    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        """Check whether the queue holds any items."""
        return len(self) == 0

    def __len__(self):
        """Return number of items in the Queue."""
        return len(self._queue)

    def __next__(self):
        """Remove and return first item in the Queue."""
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class Observable(object):
    """Inspired by https://www.protechtraining.com/blog/post/879#simple-observer."""

    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    def get_subscribers(self, event):
        """Return the subscribers of an event."""
        return self._events[event]

    def subscribe(self, event, subscriber, callback=None):
        """Add subscriber to an event."""
        if callback is None:
            callback = getattr(subscriber, 'update')
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event, subscriber):
        """Remove a subscriber for a particular event."""
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event):
        """Trigger callbacks for subscribers of an event."""
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)


class BayesianOptimization(Observable):
    """Handle optimization of a target function over a specific target space.

    This class takes the function to optimize as well as the parameters bounds
    in order to find which values for the parameters yield the maximum value
    using bayesian optimization.

    Parameters
    ----------
    f: function
        Function to be maximized.

    pbounds: dict
        Dictionary with parameters names as keys and a tuple with minimum
        and maximum values.

    constraint: A ConstraintModel. Note that the names of arguments of the
        constraint function and of f need to be the same.

    random_state: int or numpy.random.RandomState, optional(default=None)
        If the value is an integer, it is used as the seed for creating a
        numpy.random.RandomState. Otherwise the random state provided is used.
        When set to None, an unseeded random state is generated.

    verbose: int, optional(default=2)
        The level of verbosity.

    bounds_transformer: DomainTransformer, optional(default=None)
        If provided, the transformation is applied to the bounds.

    allow_duplicate_points: bool, optional (default=False)
        If True, the optimizer will allow duplicate points to be registered.
        This behavior may be desired in high noise situations where repeatedly probing
        the same point will give different answers. In other situations, the acquisition
        may occasionally generate a duplicate point.

    Methods
    -------
    probe()
        Evaluates the function on the given points.
        Can be used to guide the optimizer.

    maximize()
        Tries to find the parameters that yield the maximum value for the
        given function.

    set_bounds()
        Allows changing the lower and upper searching bounds
    """

    def __init__(self,
                 f,
                 pbounds,
                 acquisition_function=None,
                 constraint=None,
                 random_state=None,
                 verbose=2,
                 bounds_transformer=None,
                 allow_duplicate_points=False):
        self._random_state = ensure_rng(random_state)
        self._allow_duplicate_points = allow_duplicate_points
        self._queue = Queue()

        if acquisition_function is None:
            if constraint is None:
                self._acquisition_function = acquisition.UpperConfidenceBound(kappa=2.576, random_state=self._random_state)
            else:
                self._acquisition_function = acquisition.ExpectedImprovement(xi=0.01, random_state=self._random_state)
        else:
            self._acquisition_function = acquisition_function

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        if constraint is None:
            # Data structure containing the function to be optimized, the
            # bounds of its domain, and a record of the evaluations we have
            # done so far
            self._space = TargetSpace(f, pbounds, random_state=random_state,
                                      allow_duplicate_points=self._allow_duplicate_points)
            self.is_constrained = False
        else:
            constraint_ = ConstraintModel(
                constraint.fun,
                constraint.lb,
                constraint.ub,
                random_state=random_state
            )
            self._space = TargetSpace(
                f,
                pbounds,
                constraint=constraint_,
                random_state=random_state,
                allow_duplicate_points=self._allow_duplicate_points
            )
            self.is_constrained = True

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            try:
                self._bounds_transformer.initialize(self._space)
            except (AttributeError, TypeError):
                raise TypeError('The transformer must be an instance of '
                                'DomainTransformer')

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    @property
    def space(self):
        """Return the target space associated with the optimizer."""
        return self._space

    @property
    def acquisition_function(self):
        """Return the acquisition function associated with the optimizer."""
        return self._acquisition_function

    @property
    def constraint(self):
        """Return the constraint associated with the optimizer, if any."""
        if self.is_constrained:
            return self._space.constraint
        return None

    @property
    def max(self):
        """Get maximum target value found and corresponding parameters.

        See `TargetSpace.max` for more information.
        """
        return self._space.max()

    @property
    def res(self):
        """Get all target values and constraint fulfillment for all parameters.

        See `TargetSpace.res` for more information.
        """
        return self._space.res()

    def register(self, params, target, constraint_value=None):
        """Register an observation with known target.

        Parameters
        ----------
        params: dict or list
            The parameters associated with the observation.

        target: float
            Value of the target function at the observation.

        constraint_value: float or None
            Value of the constraint function at the observation, if any.
        """
        self._space.register(params, target, constraint_value)
        self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params, lazy=True):
        """Evaluate the function at the given points.

        Useful to guide the optimizer.

        Parameters
        ----------
        params: dict or list
            The parameters where the optimizer will evaluate the function.

        lazy: bool, optional(default=True)
            If True, the optimizer will evaluate the points when calling
            maximize(). Otherwise it will evaluate it at the moment.
        """
        if lazy:
            self._queue.add(params)
        else:
            self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)

    def suggest(self):
        """Suggest a promising point to probe next."""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Finding argmax of the acquisition function.
        suggestion = self._acquisition_function.suggest(
            gp=self._gp,
            target_space=self._space,
            fit_gp=True
        )

        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points):
        """Ensure the queue is not empty.

        Parameters
        ----------
        init_points: int
            Number of parameters to prime the queue with.
        """
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            self._queue.add(self._space.random_sample())

    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose, self.is_constrained)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    def maximize(self, init_points=5, n_iter=25):
        r"""
        Maximize the given function over the target space.

        Parameters
        ----------
        init_points : int, optional(default=5)
            Number of iterations before the explorations starts the exploration
            for the maximum.

        n_iter: int, optional(default=25)
            Number of iterations where the method attempts to find the maximum
            value.
        """
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)

        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                x_probe = self.suggest()
                iteration += 1
            self.probe(x_probe, lazy=False)

            if self._bounds_transformer and iteration > 0:
                # The bounds transformer should only modify the bounds after
                # the init_points points (only for the true iterations)
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)

    def set_bounds(self, new_bounds):
        """Modify the bounds of the search space.

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        """Set parameters of the internal Gaussian Process Regressor."""
        self._gp.set_params(**params)
