"""Main module.

Holds the `BayesianOptimization` class, which handles the maximization of a
function over a specific target space.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from bayes_opt import acquisition
from bayes_opt.constraint import ConstraintModel
from bayes_opt.domain_reduction import DomainTransformer
from bayes_opt.event import DEFAULT_EVENTS, Events
from bayes_opt.logger import _get_default_logger
from bayes_opt.parameter import wrap_kernel
from bayes_opt.target_space import TargetSpace
from bayes_opt.util import ensure_rng

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from numpy.random import RandomState
    from numpy.typing import NDArray
    from scipy.optimize import NonlinearConstraint

    from bayes_opt.acquisition import AcquisitionFunction
    from bayes_opt.constraint import ConstraintModel
    from bayes_opt.domain_reduction import DomainTransformer
    from bayes_opt.parameter import BoundsMapping, ParamsType

    Float = np.floating[Any]


class Observable:
    """Inspired by https://www.protechtraining.com/blog/post/879#simple-observer."""

    def __init__(self, events: Iterable[Any]) -> None:
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    def get_subscribers(self, event: Any) -> Any:
        """Return the subscribers of an event."""
        return self._events[event]

    def subscribe(self, event: Any, subscriber: Any, callback: Callable[..., Any] | None = None) -> None:
        """Add subscriber to an event."""
        if callback is None:
            callback = subscriber.update
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event: Any, subscriber: Any) -> None:
        """Remove a subscriber for a particular event."""
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event: Any) -> None:
        """Trigger callbacks for subscribers of an event."""
        for callback in self.get_subscribers(event).values():
            callback(event, self)


class BayesianOptimization(Observable):
    """Handle optimization of a target function over a specific target space.

    This class takes the function to optimize as well as the parameters bounds
    in order to find which values for the parameters yield the maximum value
    using bayesian optimization.

    Parameters
    ----------
    f: function or None.
        Function to be maximized.

    pbounds: dict
        Dictionary with parameters names as keys and a tuple with minimum
        and maximum values.

    constraint: NonlinearConstraint.
        Note that the names of arguments of the constraint function and of
        f need to be the same.

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
    """

    def __init__(
        self,
        f: Callable[..., float] | None,
        pbounds: Mapping[str, tuple[float, float]],
        acquisition_function: AcquisitionFunction | None = None,
        constraint: NonlinearConstraint | None = None,
        random_state: int | RandomState | None = None,
        verbose: int = 2,
        bounds_transformer: DomainTransformer | None = None,
        allow_duplicate_points: bool = False,
    ):
        self._random_state = ensure_rng(random_state)
        self._allow_duplicate_points = allow_duplicate_points
        self._queue: deque[ParamsType] = deque()

        if acquisition_function is None:
            if constraint is None:
                self._acquisition_function = acquisition.UpperConfidenceBound(
                    kappa=2.576, random_state=self._random_state
                )
            else:
                self._acquisition_function = acquisition.ExpectedImprovement(
                    xi=0.01, random_state=self._random_state
                )
        else:
            self._acquisition_function = acquisition_function

        if constraint is None:
            # Data structure containing the function to be optimized, the
            # bounds of its domain, and a record of the evaluations we have
            # done so far
            self._space = TargetSpace(
                f, pbounds, random_state=random_state, allow_duplicate_points=self._allow_duplicate_points
            )
            self.is_constrained = False
        else:
            constraint_ = ConstraintModel(
                constraint.fun, constraint.lb, constraint.ub, random_state=random_state
            )
            self._space = TargetSpace(
                f,
                pbounds,
                constraint=constraint_,
                random_state=random_state,
                allow_duplicate_points=self._allow_duplicate_points,
            )
            self.is_constrained = True

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=wrap_kernel(Matern(nu=2.5), transform=self._space.kernel_transform),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            if not isinstance(self._bounds_transformer, DomainTransformer):
                msg = "The transformer must be an instance of DomainTransformer"
                raise TypeError(msg)
            self._bounds_transformer.initialize(self._space)

        self._sorting_warning_already_shown = False  # TODO: remove in future version
        super().__init__(events=DEFAULT_EVENTS)

    @property
    def space(self) -> TargetSpace:
        """Return the target space associated with the optimizer."""
        return self._space

    @property
    def acquisition_function(self) -> AcquisitionFunction:
        """Return the acquisition function associated with the optimizer."""
        return self._acquisition_function

    @property
    def constraint(self) -> ConstraintModel | None:
        """Return the constraint associated with the optimizer, if any."""
        if self.is_constrained:
            return self._space.constraint
        return None

    @property
    def max(self) -> dict[str, Any] | None:
        """Get maximum target value found and corresponding parameters.

        See `TargetSpace.max` for more information.
        """
        return self._space.max()

    @property
    def res(self) -> list[dict[str, Any]]:
        """Get all target values and constraint fulfillment for all parameters.

        See `TargetSpace.res` for more information.
        """
        return self._space.res()

    def register(
        self, params: ParamsType, target: float, constraint_value: float | NDArray[Float] | None = None
    ) -> None:
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
        # TODO: remove in future version
        if isinstance(params, np.ndarray) and not self._sorting_warning_already_shown:
            msg = (
                "You're attempting to register an np.ndarray. In previous versions, the optimizer internally"
                " sorted parameters by key and expected any registered array to respect this order."
                " In the current and any future version the order as given by the pbounds dictionary will be"
                " used. If you wish to retain sorted parameters, please manually sort your pbounds"
                " dictionary before constructing the optimizer."
            )
            warn(msg, stacklevel=1)
            self._sorting_warning_already_shown = True
        self._space.register(params, target, constraint_value)
        self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params: ParamsType, lazy: bool = True) -> None:
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
        # TODO: remove in future version
        if isinstance(params, np.ndarray) and not self._sorting_warning_already_shown:
            msg = (
                "You're attempting to register an np.ndarray. In previous versions, the optimizer internally"
                " sorted parameters by key and expected any registered array to respect this order."
                " In the current and any future version the order as given by the pbounds dictionary will be"
                " used. If you wish to retain sorted parameters, please manually sort your pbounds"
                " dictionary before constructing the optimizer."
            )
            warn(msg, stacklevel=1)
            self._sorting_warning_already_shown = True
            params = self._space.array_to_params(params)
        if lazy:
            self._queue.append(params)
        else:
            self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)

    def suggest(self) -> dict[str, float | NDArray[Float]]:
        """Suggest a promising point to probe next."""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample(random_state=self._random_state))

        # Finding argmax of the acquisition function.
        suggestion = self._acquisition_function.suggest(gp=self._gp, target_space=self._space, fit_gp=True)

        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points: int) -> None:
        """Ensure the queue is not empty.

        Parameters
        ----------
        init_points: int
            Number of parameters to prime the queue with.
        """
        if not self._queue and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            sample = self._space.random_sample(random_state=self._random_state)
            self._queue.append(self._space.array_to_params(sample))

    def _prime_subscriptions(self) -> None:
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose, self.is_constrained)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    def maximize(self, init_points: int = 5, n_iter: int = 25) -> None:
        r"""
        Maximize the given function over the target space.

        Parameters
        ----------
        init_points : int, optional(default=5)
            Number of random points to probe before starting the optimization.

        n_iter: int, optional(default=25)
            Number of iterations where the method attempts to find the maximum
            value.

        Warning
        -------
            The maximize loop only fits the GP when suggesting a new point to
            probe based on the acquisition function. This means that the GP may
            not be fitted on all points registered to the target space when the
            method completes. If you intend to use the GP model after the
            optimization routine, make sure to fit it manually, e.g. by calling
            ``optimizer._gp.fit(optimizer.space.params, optimizer.space.target)``.
        """
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)

        iteration = 0
        while self._queue or iteration < n_iter:
            try:
                x_probe = self._queue.popleft()
            except IndexError:
                x_probe = self.suggest()
                iteration += 1
            self.probe(x_probe, lazy=False)

            if self._bounds_transformer and iteration > 0:
                # The bounds transformer should only modify the bounds after
                # the init_points points (only for the true iterations)
                self.set_bounds(self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)

    def set_bounds(self, new_bounds: BoundsMapping) -> None:
        """Modify the bounds of the search space.

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params: Any) -> None:
        """Set parameters of the internal Gaussian Process Regressor."""
        if "kernel" in params:
            params["kernel"] = wrap_kernel(kernel=params["kernel"], transform=self._space.kernel_transform)
        self._gp.set_params(**params)
