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
        if random_state is not None:
            msg = (
                "Providing a random_state to an acquisition function during initialization is deprecated "
                "and will be ignored. The random_state is instead provided automatically during the "
                "suggest() call."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
        self.i = 0

    @abc.abstractmethod
    def base_acq(self, *args: Any, **kwargs: Any) -> NDArray[Float]:
        """Provide access to the base acquisition function."""

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
        """Suggest a promising point to probe next.

        Parameters
        ----------
        gp : GaussianProcessRegressor
            A fitted Gaussian Process.

        target_space : TargetSpace
            The target space to probe.

        n_random : int, default 10_000
            Number of random samples to use.

        n_smart : int, default 10
            Controls the number of runs for the smart optimization. If all parameters are continuous,
            this is the number of random starting points for the L-BFGS-B optimizer. If there are
            discrete parameters, n_smart of the best points are used as starting points for the
            differential evolution optimizer, with the remaining points being random samples.

        fit_gp : bool, default True
            Whether to fit the Gaussian Process to the target space.
            Set to False if the GP is already fitted.

        random_state : int, RandomState, default None
            Random state to use for the optimization.

        Returns
        -------
        np.ndarray
            Suggested point to probe next.
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
        """Prepare the acquisition function for minimization.

        Transforms a base_acq Callable, which takes `mean` and `std` as
        input, into an acquisition function that only requires an array of
        parameters.
        Handles GP predictions and constraints.

        Parameters
        ----------
        gp : GaussianProcessRegressor
            A fitted Gaussian Process.

        constraint : ConstraintModel, default None
            A fitted constraint model, if constraints are present and the
            acquisition function supports them.

        Returns
        -------
        Callable
            Function to minimize.
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
        """Find the maximum of the acquisition function.

        Uses a combination of random sampling (cheap) and either 'L-BFGS-B' or differential evolution
        optimization (smarter, but expensive). First samples `n_random` (1e5) points at random, then
        uses the best points as starting points for the smart optimizer.

        Parameters
        ----------
        acq : Callable
            Acquisition function to use. Should accept an array of parameters `x`.

        space : TargetSpace
            The target space over which to optimize.

        random_state : RandomState
            Random state to use for the optimization.

        n_random : int
            Number of random samples to use.

        n_smart : int
            Controls the number of runs for the smart optimization. If all parameters are continuous,
            this is the number of random starting points for the L-BFGS-B optimizer. Otherwise, n_smart
            of the best points are used as starting points for the differential evolution optimizer, with
            the remaining points being random samples.

        Returns
        -------
        np.ndarray
            Parameters maximizing the acquisition function.

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
        """Random search to find the minimum of `acq` function.

        Parameters
        ----------
        acq : Callable
            Acquisition function to use. Should accept an array of parameters `x`.

        space : TargetSpace
            The target space over which to optimize.

        random_state : RandomState
            Random state to use for the optimization.

        n_random : int
            Number of random samples to use.

        n_x_seeds : int
            Number of top points to return, for use as starting points for L-BFGS-B.

        Returns
        -------
        x_min : np.ndarray
            Random sample minimizing the acquisition function.

        min_acq : float
            Acquisition function value at `x_min`
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
        """Random search to find the minimum of `acq` function.

        Parameters
        ----------
        acq : Callable
            Acquisition function to use. Should accept an array of parameters `x`.

        space : TargetSpace
            The target space over which to optimize.

        x_seeds : np.ndarray
            Starting points for the smart optimizer.
            If all parameters are continuous, this is the number of random starting points for the L-BFGS-B
            optimizer. Otherwise, n_smart of the best points are used as starting points for the differential
            evolution optimizer, with the remaining points being random samples.

        random_state : RandomState
            Random state to use for the optimization.

        Returns
        -------
        x_min : np.ndarray
            Minimal result of the L-BFGS-B optimizer.

        min_acq : float
            Acquisition function value at `x_min`
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
        if kappa < 0:
            error_msg = "kappa must be greater than or equal to 0."
            raise ValueError(error_msg)
        if exploration_decay is not None and not (0 < exploration_decay <= 1):
            error_msg = "exploration_decay must be greater than 0 and less than or equal to 1."
            raise ValueError(error_msg)
        if exploration_decay_delay is not None and (
            not isinstance(exploration_decay_delay, int) or exploration_decay_delay < 0
        ):
            error_msg = "exploration_decay_delay must be an integer greater than or equal to 0."
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
        """Suggest a promising point to probe next.

        Parameters
        ----------
        gp : GaussianProcessRegressor
            A fitted Gaussian Process.

        target_space : TargetSpace
            The target space to probe.

        n_random : int, default 10_000
            Number of random samples to use.

        n_smart : int, default 10
            Number of starting points for the L-BFGS-B optimizer.

        fit_gp : bool, default True
            Whether to fit the Gaussian Process to the target space.
            Set to False if the GP is already fitted.

        random_state : int, RandomState, default None
            Random state to use for the optimization.

        Returns
        -------
        np.ndarray
            Suggested point to probe next.
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
        """Get the current acquisition function parameters.

        Returns
        -------
        dict
            Dictionary containing the current acquisition function parameters.
        """
        return {
            "kappa": self.kappa,
            "exploration_decay": self.exploration_decay,
            "exploration_decay_delay": self.exploration_decay_delay,
        }

    def set_acquisition_params(self, params: dict) -> None:
        """Set the acquisition function parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing the acquisition function parameters.
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
        if xi < 0:
            error_msg = "xi must be greater than or equal to 0."
            raise ValueError(error_msg)
        if exploration_decay is not None and not (0 < exploration_decay <= 1):
            error_msg = "exploration_decay must be greater than 0 and less than or equal to 1."
            raise ValueError(error_msg)
        if exploration_decay_delay is not None and (
            not isinstance(exploration_decay_delay, int) or exploration_decay_delay < 0
        ):
            error_msg = "exploration_decay_delay must be an integer greater than or equal to 0."
            raise ValueError(error_msg)

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
        """Suggest a promising point to probe next.

        Parameters
        ----------
        gp : GaussianProcessRegressor
            A fitted Gaussian Process.

        target_space : TargetSpace
            The target space to probe.

        n_random : int, default 10_000
            Number of random samples to use.

        n_smart : int, default 10
            Number of starting points for the L-BFGS-B optimizer.

        fit_gp : bool, default True
            Whether to fit the Gaussian Process to the target space.
            Set to False if the GP is already fitted.

        random_state : int, RandomState, default None
            Random state to use for the optimization.

        Returns
        -------
        np.ndarray
            Suggested point to probe next.
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
        """Get the current acquisition function parameters.

        Returns
        -------
        dict
            Dictionary containing the current acquisition function parameters.
        """
        return {
            "xi": self.xi,
            "exploration_decay": self.exploration_decay,
            "exploration_decay_delay": self.exploration_decay_delay,
        }

    def set_acquisition_params(self, params: dict) -> None:
        """Set the acquisition function parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing the acquisition function parameters.
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
        if xi < 0:
            error_msg = "xi must be greater than or equal to 0."
            raise ValueError(error_msg)
        if exploration_decay is not None and not (0 < exploration_decay <= 1):
            error_msg = "exploration_decay must be greater than 0 and less than or equal to 1."
            raise ValueError(error_msg)
        if exploration_decay_delay is not None and (
            not isinstance(exploration_decay_delay, int) or exploration_decay_delay < 0
        ):
            error_msg = "exploration_decay_delay must be an integer greater than or equal to 0."
            raise ValueError(error_msg)

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
        """Suggest a promising point to probe next.

        Parameters
        ----------
        gp : GaussianProcessRegressor
            A fitted Gaussian Process.

        target_space : TargetSpace
            The target space to probe.

        n_random : int, default 10_000
            Number of random samples to use.

        n_smart : int, default 10
            Number of starting points for the L-BFGS-B optimizer.

        fit_gp : bool, default True
            Whether to fit the Gaussian Process to the target space.
            Set to False if the GP is already fitted.

        random_state : int, RandomState, default None
            Random state to use for the optimization.

        Returns
        -------
        np.ndarray
            Suggested point to probe next.
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
        """Get the current acquisition function parameters.

        Returns
        -------
        dict
            Dictionary containing the current acquisition function parameters.
        """
        return {
            "xi": self.xi,
            "exploration_decay": self.exploration_decay,
            "exploration_decay_delay": self.exploration_decay_delay,
        }

    def set_acquisition_params(self, params: dict) -> None:
        """Set the acquisition function parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing the acquisition function parameters.
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
        """Create a copy of the target space.

        Parameters
        ----------
        target_space : TargetSpace
            The target space to copy.

        Returns
        -------
        TargetSpace
            A copy of the target space.
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
        """Suggest a promising point to probe next.

        Parameters
        ----------
        gp : GaussianProcessRegressor
            A fitted Gaussian Process.

        target_space : TargetSpace
            The target space to probe.

        n_random : int, default 10_000
            Number of random samples to use.

        n_smart : int, default 10
            Number of starting points for the L-BFGS-B optimizer.

        fit_gp : bool, default True
            Unused, since the GP is always fitted to the dummy target space.
            Remains for compatibility with the base class.

        random_state : int, RandomState, default None
            Random state to use for the optimization.

        Returns
        -------
        np.ndarray
            Suggested point to probe next.
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
        """Get the current acquisition function parameters.

        Returns
        -------
        dict
            Dictionary containing the current acquisition function parameters.
        """
        return {
            "dummies": [dummy.tolist() for dummy in self.dummies],
            "base_acquisition_params": self.base_acquisition.get_acquisition_params(),
            "strategy": self.strategy,
            "atol": self.atol,
            "rtol": self.rtol,
        }

    def set_acquisition_params(self, params: dict) -> None:
        """Set the acquisition function parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing the acquisition function parameters.
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
        """Raise an error, since the base acquisition function is ambiguous."""
        msg = (
            "GPHedge base acquisition function is ambiguous."
            " You may use self.base_acquisitions[i].base_acq(mean, std)"
            " to get the base acquisition function for the i-th acquisition."
        )
        raise TypeError(msg)

    def _sample_idx_from_softmax_gains(self, random_state: RandomState) -> int:
        """Sample an index weighted by the softmax of the gains.

        Parameters
        ----------
        random_state : RandomState
            Random state to use for the sampling.

        Returns
        -------
        int
            Index of the selected base acquisition function.
        """
        cumsum_softmax_g = np.cumsum(softmax(self.gains))
        r = random_state.rand()
        return np.argmax(r <= cumsum_softmax_g)  # Returns the first True value

    def _update_gains(self, gp: GaussianProcessRegressor) -> None:
        """Update the gains of the base acquisition functions.

        Parameters
        ----------
        gp : GaussianProcessRegressor
            A fitted Gaussian Process.
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
        """Suggest a promising point to probe next.

        Parameters
        ----------
        gp : GaussianProcessRegressor
            A fitted Gaussian Process.

        target_space : TargetSpace
            The target space to probe.

        n_random : int, default 10_000
            Number of random samples to use.

        n_smart : int, default 10
            Number of starting points for the L-BFGS-B optimizer.

        fit_gp : bool, default True
            Whether to fit the Gaussian Process to the target space.
            Set to False if the GP is already fitted.

        random_state : int, RandomState, default None
            Random state to use for the optimization.

        Returns
        -------
        np.ndarray
            Suggested point to probe next.
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
        """Get the current acquisition function parameters.

        Returns
        -------
        dict
            Dictionary containing the current acquisition function parameters.
        """
        return {
            "base_acquisitions_params": [acq.get_acquisition_params() for acq in self.base_acquisitions],
            "gains": self.gains.tolist(),
            "previous_candidates": self.previous_candidates.tolist()
            if self.previous_candidates is not None
            else None,
        }

    def set_acquisition_params(self, params: dict) -> None:
        """Set the acquisition function parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing the acquisition function parameters.
        """
        for acq, acq_params in zip(self.base_acquisitions, params["base_acquisitions_params"]):
            acq.set_acquisition_params(acq_params)

        self.gains = np.array(params["gains"])
        self.previous_candidates = (
            np.array(params["previous_candidates"]) if params["previous_candidates"] is not None else None
        )
