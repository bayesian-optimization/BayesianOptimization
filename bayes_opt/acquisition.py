"""Acquisition functions for Bayesian Optimization.

The acquisition functions in this module can be grouped the following way:

- One of the base acquisition functions
  (:py:class:`UpperConfidenceBound<bayes_opt.acquisition.UpperConfidenceBound>`,
  :py:class:`ProbabilityOfImprovement<bayes_opt.acquisition.ProbabilityOfImprovement>` and
  :py:class:`ExpectedImprovement<bayes_opt.acquisition.ExpectedImprovement>`) is always dictating the basic
  behavior of the suggestion step. They can be used alone or combined with a meta acquisition function.
- :py:class:`GPHedge<bayes_opt.acquisition.GPHedge>` is a meta acquisition function that combines multiple
  base acquisition functions and determines the most suitable one for a particular problem.
- :py:class:`ConstantLiar<bayes_opt.acquisition.ConstantLiar>` is a meta acquisition function that can be
  used for parallelized optimization and discourages sampling near a previously suggested, but not yet
  evaluated, point.
- :py:class:`AcquisitionFunction<bayes_opt.acquisition.AcquisitionFunction>` is the base class for all
  acquisition functions. You can implement your own acquisition function by subclassing it. See the
  `Acquisition Functions notebook <../acquisition.html>`__ to understand the many ways this class can be
  modified.
"""

from __future__ import annotations

import abc
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import numpy as np
from numpy.random import RandomState
from packaging import version
from scipy import __version__ as scipy_version
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver, minimize
from scipy.special import softmax
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from bayes_opt.exception import (
    ConstraintNotSupportedError,
    NoValidPointRegisteredError,
    TargetSpaceEmptyError,
)
from bayes_opt.target_space import TargetSpace
from bayes_opt.util import ensure_rng

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray
    from scipy.optimize import OptimizeResult

    from bayes_opt.constraint import ConstraintModel

    Float = np.floating[Any]


class AcquisitionFunction(abc.ABC):
    """Base class for acquisition functions.

    Parameters
    ----------
    random_state : int, RandomState, default None
        Set the random state for reproducibility.
    """

    def __init__(self, random_state: int | RandomState | None = None) -> None:
        """
        Initializes the acquisition function and issues a deprecation warning if a random state is provided.
        
        Args:
            random_state: Deprecated. If provided, a warning is issued and the value is ignored.
        """
        if random_state is not None:
            msg = (
                "Providing a random_state to an acquisition function during initialization is deprecated "
                "and will be ignored. The random_state should be provided during the suggest() call."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
        self.i = 0

    @abc.abstractmethod
    def base_acq(self, *args: Any, **kwargs: Any) -> NDArray[Float]:
        """
        Computes the base acquisition function value.
        
        This method should be implemented by subclasses to return the acquisition value given predictive statistics such as mean and standard deviation.
        """

    def _fit_gp(self, gp: GaussianProcessRegressor, target_space: TargetSpace) -> None:
        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(target_space.params, target_space.target)
            if target_space.constraint is not None:
                target_space.constraint.fit(target_space.params, target_space._constraint_values)

    def get_acquisition_params(self) -> dict[str, Any]:
        """
        Get the parameters of the acquisition function.

        Returns
        -------
        dict
            The parameters of the acquisition function.
        """
        error_msg = (
            "Custom AcquisitionFunction subclasses must implement their own get_acquisition_params method."
        )
        raise NotImplementedError(error_msg)

    def set_acquisition_params(self, **params) -> None:
        """
        Set the parameters of the acquisition function.

        Parameters
        ----------
        **params : dict
            The parameters of the acquisition function.
        """
        error_msg = (
            "Custom AcquisitionFunction subclasses must implement their own set_acquisition_params method."
        )
        raise NotImplementedError(error_msg)

    def suggest(
        self,
        gp: GaussianProcessRegressor,
        target_space: TargetSpace,
        n_random: int = 10_000,
        n_smart: int = 10,
        fit_gp: bool = True,
        random_state: int | RandomState | None = None,
    ) -> NDArray[Float]:
        """
        Suggests the next candidate point to evaluate by maximizing the acquisition function.
        
        Selects a promising point in the parameter space using a combination of random sampling and local optimization, optionally fitting the Gaussian Process to the current data. Raises an error if no previous samples exist in the target space.
        
        Args:
            n_random: Number of random samples for initial exploration.
            n_smart: Number of local optimization runs for refining the suggestion.
            fit_gp: If True, fits the Gaussian Process to the target space before suggestion.
            random_state: Seed or random state for reproducibility.
        
        Returns:
            The suggested parameter vector as a NumPy array.
        
        Raises:
            TargetSpaceEmptyError: If the target space contains no previous samples.
        """
        random_state = ensure_rng(random_state)
        if len(target_space) == 0:
            msg = (
                "Cannot suggest a point without previous samples. Use "
                " target_space.random_sample() to generate a point and "
                " target_space.probe(*) to evaluate it."
            )
            raise TargetSpaceEmptyError(msg)
        self.i += 1
        if fit_gp:
            self._fit_gp(gp=gp, target_space=target_space)

        acq = self._get_acq(gp=gp, constraint=target_space.constraint)
        return self._acq_min(acq, target_space, n_random=n_random, n_smart=n_smart, random_state=random_state)

    def _get_acq(
        self, gp: GaussianProcessRegressor, constraint: ConstraintModel | None = None
    ) -> Callable[[NDArray[Float]], NDArray[Float]]:
        """
        Creates a callable acquisition function for minimization, incorporating GP predictions and optional constraints.
        
        If a constraint model is provided, the acquisition function multiplies the base acquisition value by the predicted constraint satisfaction probability. The returned function takes an array of parameters and returns the negative acquisition value for use in minimization routines.
        
        Args:
            gp: A fitted Gaussian Process regressor.
            constraint: Optional fitted constraint model.
        
        Returns:
            A callable that computes the (negative) acquisition value for given parameters, accounting for constraints if provided.
        """
        dim = gp.X_train_.shape[1]
        if constraint is not None:

            def acq(x: NDArray[Float]) -> NDArray[Float]:
                x = x.reshape(-1, dim)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mean: NDArray[Float]
                    std: NDArray[Float]
                    p_constraints: NDArray[Float]
                    mean, std = gp.predict(x, return_std=True)
                    p_constraints = constraint.predict(x)
                return -1 * self.base_acq(mean, std) * p_constraints
        else:

            def acq(x: NDArray[Float]) -> NDArray[Float]:
                x = x.reshape(-1, dim)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mean: NDArray[Float]
                    std: NDArray[Float]
                    mean, std = gp.predict(x, return_std=True)
                return -1 * self.base_acq(mean, std)

        return acq

    def _acq_min(
        self,
        acq: Callable[[NDArray[Float]], NDArray[Float]],
        space: TargetSpace,
        random_state: RandomState,
        n_random: int = 10_000,
        n_smart: int = 10,
    ) -> NDArray[Float]:
        """
        Finds the parameters that maximize the acquisition function over the target space.
        
        Combines random sampling and local optimization to efficiently search for the maximum. First, it samples `n_random` random points and identifies the best candidates. Then, it refines the search using local optimization (L-BFGS-B for continuous spaces or differential evolution for mixed spaces) starting from the top candidates. Returns the parameters corresponding to the highest acquisition value found.
        
        Args:
            acq: Acquisition function to maximize.
            space: Target space for optimization.
            random_state: Random state for reproducibility.
            n_random: Number of random samples to evaluate.
            n_smart: Number of local optimization runs for refinement.
        
        Returns:
            Parameters that maximize the acquisition function.
        """
        if n_random == 0 and n_smart == 0:
            error_msg = "Either n_random or n_smart needs to be greater than 0."
            raise ValueError(error_msg)
        x_min_r, min_acq_r, x_seeds = self._random_sample_minimize(
            acq, space, random_state, n_random=max(n_random, n_smart), n_x_seeds=n_smart
        )
        if n_smart:
            x_min_s, min_acq_s = self._smart_minimize(acq, space, x_seeds=x_seeds, random_state=random_state)
            # Either n_random or n_smart is not 0 => at least one of x_min_r and x_min_s is not None
            if min_acq_r > min_acq_s:
                return x_min_s
        return x_min_r

    def _random_sample_minimize(
        self,
        acq: Callable[[NDArray[Float]], NDArray[Float]],
        space: TargetSpace,
        random_state: RandomState,
        n_random: int,
        n_x_seeds: int = 0,
    ) -> tuple[NDArray[Float] | None, float, NDArray[Float]]:
        """
        Performs random sampling to find the minimum of the acquisition function.
        
        Randomly samples points from the target space, evaluates the acquisition function at each, and returns the point with the lowest acquisition value. Optionally returns the top points for use as seeds in further optimization.
        
        Args:
            acq: Acquisition function to minimize.
            space: Target space to sample from.
            random_state: Random state for reproducibility.
            n_random: Number of random samples to draw.
            n_x_seeds: Number of top points to return as seeds.
        
        Returns:
            A tuple containing:
                - The sampled point with the lowest acquisition value (or None if n_random is 0).
                - The minimum acquisition value found.
                - An array of the top n_x_seeds points with the lowest acquisition values.
        """
        if n_random == 0:
            return None, np.inf, space.random_sample(n_x_seeds, random_state=random_state)
        x_tries = space.random_sample(n_random, random_state=random_state)
        ys = acq(x_tries)
        x_min = x_tries[ys.argmin()]
        min_acq = ys.min()
        if n_x_seeds != 0:
            idxs = np.argsort(ys)[:n_x_seeds]
            x_seeds = x_tries[idxs]
        else:
            x_seeds = []
        return x_min, min_acq, x_seeds

    def _smart_minimize(
        self,
        acq: Callable[[NDArray[Float]], NDArray[Float]],
        space: TargetSpace,
        x_seeds: NDArray[Float],
        random_state: RandomState,
    ) -> tuple[NDArray[Float] | None, float]:
        """
        Performs local optimization to find the minimum of the acquisition function.
        
        Uses L-BFGS-B for continuous parameter spaces and differential evolution for mixed-integer spaces, optionally refining continuous parameters after global optimization. Returns the best found point and its acquisition value.
        
        Args:
            acq: Acquisition function to minimize.
            space: Target space defining parameter bounds and types.
            x_seeds: Initial points for local or global optimization.
            random_state: Random state for reproducibility.
        
        Returns:
            A tuple containing the best found point and its acquisition function value.
        """
        continuous_dimensions = space.continuous_dimensions
        continuous_bounds = space.bounds[continuous_dimensions]

        min_acq: float | None = None
        x_try: NDArray[Float]
        x_min: NDArray[Float]

        # Case of continous optimization
        if all(continuous_dimensions):
            for x_try in x_seeds:
                res: OptimizeResult = minimize(acq, x_try, bounds=continuous_bounds, method="L-BFGS-B")
                if not res.success:
                    continue

                # Store it if better than previous minimum(maximum).
                if min_acq is None or np.squeeze(res.fun) < min_acq:
                    x_try = res.x
                    x_min = x_try
                    min_acq = np.squeeze(res.fun)
        # Case of mixed-integer optimization
        else:
            xinit = space.random_sample(15 * len(space.bounds), random_state=random_state)
            if len(x_seeds) > 0:
                n_seeds = min(len(x_seeds), len(xinit))
                xinit[:n_seeds] = x_seeds[:n_seeds]

            de_parameters = {"func": acq, "bounds": space.bounds, "polish": False, "init": xinit}
            if version.parse(scipy_version) < version.parse("1.15.0"):
                de_parameters["seed"] = random_state
            else:
                de_parameters["rng"] = random_state

            de = DifferentialEvolutionSolver(**de_parameters)
            res_de: OptimizeResult = de.solve()
            # Check if success
            if not res_de.success:
                msg = f"Differential evolution optimization failed. Message: {res_de.message}"
                raise RuntimeError(msg)

            x_min = res_de.x
            min_acq = np.squeeze(res_de.fun)

            # Refine the identification of continous parameters with deterministic search
            if any(continuous_dimensions):
                x_try = x_min.copy()

                def continuous_acq(x: NDArray[Float], x_try=x_try) -> NDArray[Float]:
                    """
                    Evaluates the acquisition function at a point with updated continuous dimensions.
                    
                    Args:
                        x: Array of values for the continuous dimensions.
                    
                    Returns:
                        The acquisition function value at the point formed by replacing the continuous dimensions of `x_try` with `x`.
                    """
                    x_try[continuous_dimensions] = x
                    return acq(x_try)

                res: OptimizeResult = minimize(
                    continuous_acq, x_min[continuous_dimensions], bounds=continuous_bounds
                )
                if res.success and np.squeeze(res.fun) < min_acq:
                    x_try[continuous_dimensions] = res.x
                    x_min = x_try
                    min_acq = np.squeeze(res.fun)

        if min_acq is None:
            min_acq = np.inf
            x_min = np.array([np.nan] * space.bounds.shape[0])

        # Clip output to make sure it lies within the bounds. Due to floating
        # point technicalities this is not always the case.
        return np.clip(x_min, space.bounds[:, 0], space.bounds[:, 1]), min_acq


class UpperConfidenceBound(AcquisitionFunction):
    r"""Upper Confidence Bound acquisition function.

    The upper confidence bound is calculated as

    .. math::
        \text{UCB}(x) = \mu(x) + \kappa \sigma(x).

    Parameters
    ----------
    kappa : float, default 2.576
        Governs the exploration/exploitation tradeoff. Lower prefers
        exploitation, higher prefers exploration.

    exploration_decay : float, default None
        Decay rate for kappa. If None, no decay is applied.

    exploration_decay_delay : int, default None
        Delay for decay. If None, decay is applied from the start.

    """

    def __init__(
        self,
        kappa: float = 2.576,
        exploration_decay: float | None = None,
        exploration_decay_delay: int | None = None,
        random_state: int | RandomState | None = None,
    ) -> None:
        """
        Initializes the Upper Confidence Bound (UCB) acquisition function.
        
        Args:
            kappa: Controls the exploration/exploitation tradeoff; must be non-negative.
            exploration_decay: Optional decay rate for kappa after each suggestion.
            exploration_decay_delay: Optional delay (in iterations) before decay starts.
            random_state: Deprecated. If provided, triggers a warning; use random_state in suggest() instead.
        
        Raises:
            ValueError: If kappa is negative.
        """
        if kappa < 0:
            error_msg = "kappa must be greater than or equal to 0."
            raise ValueError(error_msg)

        super().__init__(random_state=random_state)
        self.kappa = kappa
        self.exploration_decay = exploration_decay
        self.exploration_decay_delay = exploration_decay_delay

    def base_acq(self, mean: NDArray[Float], std: NDArray[Float]) -> NDArray[Float]:
        """Calculate the upper confidence bound.

        Parameters
        ----------
        mean : np.ndarray
            Mean of the predictive distribution.

        std : np.ndarray
            Standard deviation of the predictive distribution.

        Returns
        -------
        np.ndarray
            Acquisition function value.
        """
        return mean + self.kappa * std

    def suggest(
        self,
        gp: GaussianProcessRegressor,
        target_space: TargetSpace,
        n_random: int = 10_000,
        n_smart: int = 10,
        fit_gp: bool = True,
        random_state: int | RandomState | None = None,
    ) -> NDArray[Float]:
        """
        Suggests the next point to evaluate using the Upper Confidence Bound acquisition function.
        
        Raises:
            ConstraintNotSupportedError: If constraints are present in the target space.
            
        Returns:
            The suggested point as a NumPy array.
        """
        if target_space.constraint is not None:
            msg = (
                f"Received constraints, but acquisition function {type(self)} "
                "does not support constrained optimization."
            )
            raise ConstraintNotSupportedError(msg)
        x_max = super().suggest(
            gp=gp,
            target_space=target_space,
            n_random=n_random,
            n_smart=n_smart,
            fit_gp=fit_gp,
            random_state=random_state,
        )
        self.decay_exploration()
        return x_max

    def decay_exploration(self) -> None:
        """Decay kappa by a constant rate.

        Adjust exploration/exploitation trade-off by reducing kappa.

        Note
        ----

        This method is called automatically at the end of each ``suggest()`` call.
        """
        if self.exploration_decay is not None and (
            self.exploration_decay_delay is None or self.exploration_decay_delay <= self.i
        ):
            self.kappa = self.kappa * self.exploration_decay

    def get_acquisition_params(self) -> dict:
        """
        Returns the current parameters of the Upper Confidence Bound acquisition function.
        
        Returns:
            A dictionary with keys 'kappa', 'exploration_decay', and 'exploration_decay_delay'
            representing the current values of these parameters.
        """
        return {
            "kappa": self.kappa,
            "exploration_decay": self.exploration_decay,
            "exploration_decay_delay": self.exploration_decay_delay,
        }

    def set_acquisition_params(self, params: dict) -> None:
        """
        Sets the parameters for the Upper Confidence Bound acquisition function.
        
        Args:
            params: Dictionary with keys "kappa", "exploration_decay", and "exploration_decay_delay".
        """
        self.kappa = params["kappa"]
        self.exploration_decay = params["exploration_decay"]
        self.exploration_decay_delay = params["exploration_decay_delay"]


class ProbabilityOfImprovement(AcquisitionFunction):
    r"""Probability of Improvement acqusition function.

    Calculated as

    .. math:: \text{POI}(x) = \Phi\left( \frac{\mu(x)-y_{\text{max}} -  \xi }{\sigma(x)} \right)

    where :math:`\Phi` is the CDF of the normal distribution.

    Parameters
    ----------
    xi : float, positive
        Governs the exploration/exploitation tradeoff. Lower prefers
        exploitation, higher prefers exploration.

    exploration_decay : float, default None
        Decay rate for xi. If None, no decay is applied.

    exploration_decay_delay : int, default None
        Delay for decay. If None, decay is applied from the start.

    random_state : int, RandomState, default None
        Set the random state for reproducibility.
    """

    def __init__(
        self,
        xi: float,
        exploration_decay: float | None = None,
        exploration_decay_delay: int | None = None,
        random_state: int | RandomState | None = None,
    ) -> None:
        super().__init__(random_state=random_state)
        self.xi = xi
        self.exploration_decay = exploration_decay
        self.exploration_decay_delay = exploration_decay_delay
        self.y_max = None

    def base_acq(self, mean: NDArray[Float], std: NDArray[Float]) -> NDArray[Float]:
        """Calculate the probability of improvement.

        Parameters
        ----------
        mean : np.ndarray
            Mean of the predictive distribution.

        std : np.ndarray
            Standard deviation of the predictive distribution.

        Returns
        -------
        np.ndarray
            Acquisition function value.

        Raises
        ------
        ValueError
            If y_max is not set.
        """
        if self.y_max is None:
            msg = (
                "y_max is not set. If you are calling this method outside "
                "of suggest(), you must set y_max manually."
            )
            raise ValueError(msg)
        z = (mean - self.y_max - self.xi) / std
        return norm.cdf(z)

    def suggest(
        self,
        gp: GaussianProcessRegressor,
        target_space: TargetSpace,
        n_random: int = 10_000,
        n_smart: int = 10,
        fit_gp: bool = True,
        random_state: int | RandomState | None = None,
    ) -> NDArray[Float]:
        """
        Suggests the next point to evaluate using the Expected Improvement acquisition function.
        
        Raises:
            NoValidPointRegisteredError: If no valid points satisfying constraints are available in the target space.
        
        Returns:
            The suggested point as a NumPy array.
        """
        y_max = target_space._target_max()
        if y_max is None and not target_space.empty:
            # If target space is empty, let base class handle the error
            msg = (
                "Cannot suggest a point without an allowed point. Use "
                "target_space.random_sample() to generate a point until "
                " at least one point that satisfies the constraints is found."
            )
            raise NoValidPointRegisteredError(msg)
        self.y_max = y_max
        x_max = super().suggest(
            gp=gp,
            target_space=target_space,
            n_random=n_random,
            n_smart=n_smart,
            fit_gp=fit_gp,
            random_state=random_state,
        )
        self.decay_exploration()
        return x_max

    def decay_exploration(self) -> None:
        r"""Decay xi by a constant rate.

        Adjust exploration/exploitation trade-off by reducing xi.

        Note
        ----

        This method is called automatically at the end of each ``suggest()`` call.
        """
        if self.exploration_decay is not None and (
            self.exploration_decay_delay is None or self.exploration_decay_delay <= self.i
        ):
            self.xi = self.xi * self.exploration_decay

    def get_acquisition_params(self) -> dict:
        """
        Returns the current parameters of the acquisition function as a dictionary.
        
        The dictionary includes the exploration parameter `xi`, the decay rate, and the decay delay.
        """
        return {
            "xi": self.xi,
            "exploration_decay": self.exploration_decay,
            "exploration_decay_delay": self.exploration_decay_delay,
        }

    def set_acquisition_params(self, params: dict) -> None:
        """
        Sets the acquisition function parameters from a dictionary.
        
        Args:
            params: Dictionary with keys "xi", "exploration_decay", and "exploration_decay_delay" specifying the acquisition function's configuration.
        """
        self.xi = params["xi"]
        self.exploration_decay = params["exploration_decay"]
        self.exploration_decay_delay = params["exploration_decay_delay"]


class ExpectedImprovement(AcquisitionFunction):
    r"""Expected Improvement acqusition function.

    Similar to Probability of Improvement (`ProbabilityOfImprovement`), but also considers the
    magnitude of improvement.
    Calculated as

    .. math::
        \text{EI}(x) = (\mu(x)-y_{\text{max}} - \xi) \Phi\left(
            \frac{\mu(x)-y_{\text{max}} -  \xi }{\sigma(x)} \right)
                + \sigma(x) \phi\left(
                \frac{\mu(x)-y_{\text{max}} -  \xi }{\sigma(x)} \right)

    where :math:`\Phi` is the CDF and :math:`\phi` the PDF of the normal
    distribution.

    Parameters
    ----------
    xi : float, positive
        Governs the exploration/exploitation tradeoff. Lower prefers
        exploitation, higher prefers exploration.

    exploration_decay : float, default None
        Decay rate for xi. If None, no decay is applied.

    exploration_decay_delay : int, default None

    random_state : int, RandomState, default None
        Set the random state for reproducibility.
    """

    def __init__(
        self,
        xi: float,
        exploration_decay: float | None = None,
        exploration_decay_delay: int | None = None,
        random_state: int | RandomState | None = None,
    ) -> None:
        super().__init__(random_state=random_state)
        self.xi = xi
        self.exploration_decay = exploration_decay
        self.exploration_decay_delay = exploration_decay_delay
        self.y_max = None

    def base_acq(self, mean: NDArray[Float], std: NDArray[Float]) -> NDArray[Float]:
        """Calculate the expected improvement.

        Parameters
        ----------
        mean : np.ndarray
            Mean of the predictive distribution.

        std : np.ndarray
            Standard deviation of the predictive distribution.

        Returns
        -------
        np.ndarray
            Acquisition function value.

        Raises
        ------
        ValueError
            If y_max is not set.
        """
        if self.y_max is None:
            msg = (
                "y_max is not set. If you are calling this method outside "
                "of suggest(), ensure y_max is set, or set it manually."
            )
            raise ValueError(msg)
        a = mean - self.y_max - self.xi
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    def suggest(
        self,
        gp: GaussianProcessRegressor,
        target_space: TargetSpace,
        n_random: int = 10_000,
        n_smart: int = 10,
        fit_gp: bool = True,
        random_state: int | RandomState | None = None,
    ) -> NDArray[Float]:
        """
        Suggests the next point to evaluate using the Expected Improvement acquisition function.
        
        Raises:
            NoValidPointRegisteredError: If no valid points satisfying constraints are available in the target space.
        
        Returns:
            The next suggested point as a NumPy array.
        """
        y_max = target_space._target_max()
        if y_max is None and not target_space.empty:
            # If target space is empty, let base class handle the error
            msg = (
                "Cannot suggest a point without an allowed point. Use "
                "target_space.random_sample() to generate a point until "
                " at least one point that satisfies the constraints is found."
            )
            raise NoValidPointRegisteredError(msg)
        self.y_max = y_max

        x_max = super().suggest(
            gp=gp,
            target_space=target_space,
            n_random=n_random,
            n_smart=n_smart,
            fit_gp=fit_gp,
            random_state=random_state,
        )
        self.decay_exploration()
        return x_max

    def decay_exploration(self) -> None:
        r"""Decay xi by a constant rate.

        Adjust exploration/exploitation trade-off by reducing xi.

        Note
        ----

        This method is called automatically at the end of each ``suggest()`` call.
        """
        if self.exploration_decay is not None and (
            self.exploration_decay_delay is None or self.exploration_decay_delay <= self.i
        ):
            self.xi = self.xi * self.exploration_decay

    def get_acquisition_params(self) -> dict:
        """
        Returns the current parameters of the acquisition function as a dictionary.
        
        The dictionary includes the exploration parameter `xi`, the decay rate, and the decay delay.
        """
        return {
            "xi": self.xi,
            "exploration_decay": self.exploration_decay,
            "exploration_decay_delay": self.exploration_decay_delay,
        }

    def set_acquisition_params(self, params: dict) -> None:
        """
        Sets the acquisition function parameters from a dictionary.
        
        Args:
            params: Dictionary with keys "xi", "exploration_decay", and "exploration_decay_delay" specifying the acquisition function's configuration.
        """
        self.xi = params["xi"]
        self.exploration_decay = params["exploration_decay"]
        self.exploration_decay_delay = params["exploration_decay_delay"]


class ConstantLiar(AcquisitionFunction):
    """Constant Liar acquisition function.

    Used for asynchronous optimization. It operates on a copy of the target space
    that includes the previously suggested points that have not been evaluated yet.
    A GP fitted to this target space is less likely to suggest the same point again,
    since the variance of the predictive distribution is lower at these points.
    This is discourages the optimization algorithm from suggesting the same point
    to multiple workers.

    Parameters
    ----------
    base_acquisition : AcquisitionFunction
        The acquisition function to use.

    strategy : float or str, default 'max'
        Strategy to use for the constant liar. If a float, the constant liar
        will always register dummies with this value. If 'min'/'mean'/'max',
        the constant liar will register dummies with the minimum/mean/maximum
        target value in the target space.

    random_state : int, RandomState, default None
        Set the random state for reproducibility.

    atol : float, default 1e-5
        Absolute tolerance to eliminate a dummy point.

    rtol : float, default 1e-8
        Relative tolerance to eliminate a dummy point.
    """

    def __init__(
        self,
        base_acquisition: AcquisitionFunction,
        strategy: Literal["min", "mean", "max"] | float = "max",
        random_state: int | RandomState | None = None,
        atol: float = 1e-5,
        rtol: float = 1e-8,
    ) -> None:
        super().__init__(random_state)
        self.base_acquisition = base_acquisition
        self.dummies = []
        if not isinstance(strategy, float) and strategy not in ["min", "mean", "max"]:
            error_msg = f"Received invalid argument {strategy} for strategy."
            raise ValueError(error_msg)
        self.strategy: Literal["min", "mean", "max"] | float = strategy
        self.atol = atol
        self.rtol = rtol

    def base_acq(self, *args: Any, **kwargs: Any) -> NDArray[Float]:
        """Calculate the acquisition function.

        Calls the base acquisition function's `base_acq` method.

        Returns
        -------
        np.ndarray
            Acquisition function value.
        """
        return self.base_acquisition.base_acq(*args, **kwargs)

    def _copy_target_space(self, target_space: TargetSpace) -> TargetSpace:
        """
        Creates a deep copy of the given target space, including its bounds, parameters, targets, and constraints.
        
        Args:
            target_space: The target space to be copied.
        
        Returns:
            A new TargetSpace instance with duplicated data and constraints.
        """
        keys = target_space.keys
        pbounds = {key: bound for key, bound in zip(keys, target_space.bounds)}
        target_space_copy = TargetSpace(
            None, pbounds=pbounds, allow_duplicate_points=target_space._allow_duplicate_points
        )
        if target_space._constraint is not None:
            target_space_copy.set_constraint(deepcopy(target_space.constraint))

        target_space_copy._params = deepcopy(target_space._params)
        target_space_copy._target = deepcopy(target_space._target)

        return target_space_copy

    def _remove_expired_dummies(self, target_space: TargetSpace) -> None:
        """Remove expired dummy points from the list of dummies.

        Once a worker has evaluated a dummy point, the dummy is discarded. To
        accomplish this, we compare every dummy point to the current target
        space's parameters and remove it if it is close to any of them.

        Parameters
        ----------
        target_space : TargetSpace
            The target space to compare the dummies to.
        """
        dummies = []
        for dummy in self.dummies:
            close = np.isclose(dummy, target_space.params, rtol=self.rtol, atol=self.atol)
            if not close.all(axis=1).any():
                dummies.append(dummy)
        self.dummies = dummies

    def suggest(
        self,
        gp: GaussianProcessRegressor,
        target_space: TargetSpace,
        n_random: int = 10_000,
        n_smart: int = 10,
        fit_gp: bool = True,
        random_state: int | RandomState | None = None,
    ) -> NDArray[Float]:
        """
        Suggests the next point to evaluate using the Constant Liar meta acquisition strategy.
        
        Creates a copy of the target space augmented with "dummy" points representing pending evaluations, assigns them a target value based on the specified strategy, and fits the Gaussian Process to this augmented space. Returns a new candidate point by delegating to the wrapped base acquisition function. Raises an error if no previous samples exist or if constraints are present.
        
        Args:
            n_random: Number of random samples for initial exploration.
            n_smart: Number of starting points for local optimization.
            random_state: Random seed or generator for reproducibility.
        
        Returns:
            The suggested point as a NumPy array.
        """
        if len(target_space) == 0:
            msg = (
                "Cannot suggest a point without previous samples. Use "
                " target_space.random_sample() to generate a point and "
                " target_space.probe(*) to evaluate it."
            )
            raise TargetSpaceEmptyError(msg)

        if target_space.constraint is not None:
            msg = (
                f"Received constraints, but acquisition function {type(self)} "
                "does not support constrained optimization."
            )
            raise ConstraintNotSupportedError(msg)

        # Check if any dummies have been evaluated and remove them
        self._remove_expired_dummies(target_space)

        # Create a copy of the target space
        dummy_target_space = self._copy_target_space(target_space)

        dummy_target: float
        # Choose the dummy target value
        if isinstance(self.strategy, float):
            dummy_target = self.strategy
        elif self.strategy == "min":
            dummy_target = target_space.target.min()
        elif self.strategy == "mean":
            dummy_target = target_space.target.mean()
        elif self.strategy != "max":
            error_msg = f"Received invalid argument {self.strategy} for strategy."
            raise ValueError(error_msg)
        else:
            dummy_target = target_space.target.max()

        # Register the dummies to the dummy target space
        for dummy in self.dummies:
            dummy_target_space.register(dummy, dummy_target)

        # Fit the GP to the dummy target space and suggest a point
        self._fit_gp(gp=gp, target_space=dummy_target_space)
        x_max = self.base_acquisition.suggest(
            gp,
            dummy_target_space,
            n_random=n_random,
            n_smart=n_smart,
            fit_gp=False,
            random_state=random_state,
        )

        # Register the suggested point as a dummy
        self.dummies.append(x_max)

        return x_max

    def get_acquisition_params(self) -> dict:
        """
        Returns a dictionary of the current ConstantLiar acquisition function parameters.
        
        The dictionary includes the list of dummy points, the parameters of the wrapped base acquisition function, the dummy value strategy, and tolerance settings.
        """
        return {
            "dummies": [dummy.tolist() for dummy in self.dummies],
            "base_acquisition_params": self.base_acquisition.get_acquisition_params(),
            "strategy": self.strategy,
            "atol": self.atol,
            "rtol": self.rtol,
        }

    def set_acquisition_params(self, params: dict) -> None:
        """
        Sets the acquisition function parameters from a dictionary.
        
        Updates the dummy points, base acquisition parameters, strategy, and tolerances based on the provided dictionary.
        """
        self.dummies = [np.array(dummy) for dummy in params["dummies"]]
        self.base_acquisition.set_acquisition_params(params["base_acquisition_params"])
        self.strategy = params["strategy"]
        self.atol = params["atol"]
        self.rtol = params["rtol"]


class GPHedge(AcquisitionFunction):
    """GPHedge acquisition function.

    At each suggestion step, GPHedge samples suggestions from each base
    acquisition function acq_i. Then a candidate is selected from the
    suggestions based on the on the cumulative rewards of each acq_i.
    After evaluating the candidate, the gains are updated (in the next
    iteration) based on the updated expectation value of the candidates.

    For more information, see:
        Brochu et al., "Portfolio Allocation for Bayesian Optimization",
        https://arxiv.org/abs/1009.5419

    Parameters
    ----------
    base_acquisitions : Sequence[AcquisitionFunction]
        Sequence of base acquisition functions.

    random_state : int, RandomState, default None
        Set the random state for reproducibility.
    """

    def __init__(
        self, base_acquisitions: Sequence[AcquisitionFunction], random_state: int | RandomState | None = None
    ) -> None:
        super().__init__(random_state)
        self.base_acquisitions = list(base_acquisitions)
        self.n_acq = len(self.base_acquisitions)
        self.gains = np.zeros(self.n_acq)
        self.previous_candidates = None

    def base_acq(self, *args: Any, **kwargs: Any) -> NoReturn:
        """
        Raises an error because the base acquisition function is not defined for GPHedge.
        
        Calling this method is invalid, as GPHedge combines multiple acquisition functions and does not have a single base acquisition function. Use the base acquisition function of a specific component instead.
        """
        msg = (
            "GPHedge base acquisition function is ambiguous."
            " You may use self.base_acquisitions[i].base_acq(mean, std)"
            " to get the base acquisition function for the i-th acquisition."
        )
        raise TypeError(msg)

    def _sample_idx_from_softmax_gains(self, random_state: RandomState) -> int:
        """
        Samples an index according to the softmax-weighted probabilities of the current gains.
        
        Args:
            random_state: Random state used for reproducible sampling.
        
        Returns:
            The index of the selected base acquisition function, sampled proportionally to its softmax gain.
        """
        cumsum_softmax_g = np.cumsum(softmax(self.gains))
        r = random_state.rand()
        return np.argmax(r <= cumsum_softmax_g)  # Returns the first True value

    def _update_gains(self, gp: GaussianProcessRegressor) -> None:
        """
        Updates the cumulative gains for each base acquisition function using the predicted rewards of previous candidates.
        
        Args:
            gp: A fitted GaussianProcessRegressor used to predict rewards for previous candidates.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rewards = gp.predict(self.previous_candidates)
        self.gains += rewards
        self.previous_candidates = None

    def suggest(
        self,
        gp: GaussianProcessRegressor,
        target_space: TargetSpace,
        n_random: int = 10_000,
        n_smart: int = 10,
        fit_gp: bool = True,
        random_state: int | RandomState | None = None,
    ) -> NDArray[Float]:
        """
        Suggests the next point to evaluate by combining multiple acquisition functions using the GP-Hedge strategy.
        
        Selects a candidate from several base acquisition functions, weighted by their historical performance, to balance exploration and exploitation in Bayesian Optimization.
        
        Args:
            n_random: Number of random samples for candidate generation per acquisition function.
            n_smart: Number of local optimizer starts per acquisition function.
            fit_gp: If True, fits the Gaussian Process to the target space before suggesting.
            random_state: Seed or random state for reproducibility.
        
        Returns:
            The selected point to probe next as a NumPy array.
        
        Raises:
            TargetSpaceEmptyError: If no previous samples exist in the target space.
        """
        if len(target_space) == 0:
            msg = (
                "Cannot suggest a point without previous samples. Use "
                " target_space.random_sample() to generate a point and "
                " target_space.probe(*) to evaluate it."
            )
            raise TargetSpaceEmptyError(msg)
        self.i += 1
        random_state = ensure_rng(random_state)
        if fit_gp:
            self._fit_gp(gp=gp, target_space=target_space)

        # Update the gains of the base acquisition functions
        if self.previous_candidates is not None:
            self._update_gains(gp)

        # Suggest a point using each base acquisition function
        x_max = [
            base_acq.suggest(
                gp=gp,
                target_space=target_space,
                n_random=n_random // self.n_acq,
                n_smart=n_smart // self.n_acq,
                fit_gp=False,
                random_state=random_state,
            )
            for base_acq in self.base_acquisitions
        ]
        self.previous_candidates = np.array(x_max)
        idx = self._sample_idx_from_softmax_gains(random_state=random_state)
        return x_max[idx]

    def get_acquisition_params(self) -> dict:
        """
        Returns a dictionary of the current parameters, gains, and previous candidates for the GPHedge acquisition function.
        
        Returns
        -------
        dict
            A dictionary with keys 'base_acquisitions_params', 'gains', and 'previous_candidates' representing the state of the GPHedge acquisition function.
        """
        return {
            "base_acquisitions_params": [acq.get_acquisition_params() for acq in self.base_acquisitions],
            "gains": self.gains.tolist(),
            "previous_candidates": self.previous_candidates.tolist()
            if self.previous_candidates is not None
            else None,
        }

    def set_acquisition_params(self, params: dict) -> None:
        """
        Sets the parameters for the GPHedge acquisition function and its base acquisition functions.
        
        Args:
            params: Dictionary containing serialized parameters for each base acquisition, cumulative gains, and previous candidates.
        """
        for acq, acq_params in zip(self.base_acquisitions, params["base_acquisitions_params"]):
            acq.set_acquisition_params(acq_params)

        self.gains = np.array(params["gains"])
        self.previous_candidates = (
            np.array(params["previous_candidates"]) if params["previous_candidates"] is not None else None
        )
