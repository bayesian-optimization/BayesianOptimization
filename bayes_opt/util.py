"""Contains utility functions."""
import json
import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def acq_max(ac, gp, y_max, bounds, random_state, constraint=None, n_warmup=10000, n_iter=10, y_max_params=None):
    """Find the maximum of the acquisition function.

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (10) random starting points.

    Parameters
    ----------
    ac : callable
        Acquisition function to use. Should accept an array of parameters `x`,
        an from sklearn.gaussian_process.GaussianProcessRegressor `gp` and the
        best current value `y_max` as parameters.
        
    gp : sklearn.gaussian_process.GaussianProcessRegressor
        A gaussian process regressor modelling the target function based on
        previous observations.
        
    y_max : number
        Highest found value of the target function.

    bounds : np.ndarray
        Bounds of the search space. For `N` parameters this has shape
        `(N, 2)` with `[i, 0]` the lower bound of parameter `i` and
        `[i, 1]` the upper bound.
        
    random_state : np.random.RandomState
        A random state to sample from.

    constraint : ConstraintModel or None, default=None
        If provided, the acquisition function will be adjusted according
        to the probability of fulfilling the constraint.

    n_warmup : int, default=10000
        Number of points to sample from the acquisition function as seeds
        before looking for a minimum.

    n_iter : int, default=10
        Points to run L-BFGS-B optimization from.

    y_max_params : np.array
        Function parameters that produced the maximum known value given by `y_max`.

    :param y_max_params:
        Function parameters that produced the maximum known value given by `y_max`.

    Returns
    -------
    Parameters maximizing the acquisition function.

    """
    # We need to adjust the acquisition function to deal with constraints when there is some
    if constraint is not None:
        def adjusted_ac(x):
            """Acquisition function adjusted to fulfill the constraint when necessary.

            Parameters
            ----------
            x : np.ndarray
                Parameter at which to sample.


            Returns
            -------
            The value of the acquisition function adjusted for constraints.
            """
            # Transforms the problem in a minimization problem, this is necessary
            # because the solver we are using later on is a minimizer
            values = -ac(x.reshape(-1, bounds.shape[0]), gp=gp, y_max=y_max)
            p_constraints = constraint.predict(x.reshape(-1, bounds.shape[0]))

            # Slower fallback for the case where any values are negative
            if np.any(values > 0):
                # TODO: This is not exactly how Gardner et al do it.
                # Their way would require the result of the acquisition function
                # to be strictly positive, which is not the case here. For a
                # positive target value, we use Gardner's version. If the target
                # is negative, we instead slightly rescale the target depending
                # on the probability estimate to fulfill the constraint.
                return np.array(
                    [
                        value / (0.5 + 0.5 * p) if value > 0 else value * p
                        for value, p in zip(values, p_constraints)
                    ]
                )

            # Faster, vectorized version of Gardner et al's method
            return values * p_constraints

    else:
        # Transforms the problem in a minimization problem, this is necessary
        # because the solver we are using later on is a minimizer
        adjusted_ac = lambda x: -ac(x.reshape(-1, bounds.shape[0]), gp=gp, y_max=y_max)

    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    ys = -adjusted_ac(x_tries)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more thoroughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(1+n_iter+int(not y_max_params is None),
                                   bounds.shape[0]))
    # Add the best candidate from the random sampling to the seeds so that the
    # optimization algorithm can try to walk up to that particular local maxima
    x_seeds[0] = x_max
    if not y_max_params is None:
        # Add the provided best sample to the seeds so that the optimization
        # algorithm is aware of it and will attempt to find its local maxima
        x_seeds[1] = y_max_params

    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(adjusted_ac,
                       x_try,
                       bounds=bounds,
                       method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -np.squeeze(res.fun) >= max_acq:
            x_max = res.x
            max_acq = -np.squeeze(res.fun)

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


class UtilityFunction():
    """An object to compute the acquisition functions.
    
    Parameters
    ----------
    kind: {'ucb', 'ei', 'poi'}
        * 'ucb' stands for the Upper Confidence Bounds method
        * 'ei' is the Expected Improvement method
        * 'poi' is the Probability Of Improvement criterion.
    
    kappa: float, optional(default=2.576)
            Parameter to indicate how closed are the next parameters sampled.
            Higher value = favors spaces that are least explored.
            Lower value = favors spaces where the regression function is
            the highest.
    
    kappa_decay: float, optional(default=1)
        `kappa` is multiplied by this factor every iteration.
    
    kappa_decay_delay: int, optional(default=0)
        Number of iterations that must have passed before applying the
        decay to `kappa`.
    
    xi: float, optional(default=0.0)
    """

    def __init__(self, kind='ucb', kappa=2.576, xi=0, kappa_decay=1, kappa_decay_delay=0):
        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.xi = xi

        self._iters_counter = 0

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  f"{kind} has not been implemented, " \
                  "please choose one of ucb, ei, or poi."
            raise NotImplementedError(err)
        self.kind = kind

    def update_params(self):
        """Update internal parameters."""
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max):
        """Calculate acquisition function.

        Parameters
        ----------
        x : np.ndarray
            Parameters to evaluate the function at.

        gp : sklearn.gaussian_process.GaussianProcessRegressor
            A gaussian process regressor modelling the target function based on
            previous observations.

        y_max : number
            Highest found value of the target function.


        Returns
        -------
        Values of the acquisition function
        """
        if self.kind == 'ucb':
            return self.ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self.ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self.poi(x, gp, y_max, self.xi)
        raise ValueError(f"{self.kind} is not a valid acquisition function.")

    @staticmethod
    def ucb(x, gp, kappa):
        r"""Calculate Upper Confidence Bound acquisition function.

        Similar to Probability of Improvement (`UtilityFunction.poi`), but also considers the
        magnitude of improvement.
        Calculated as
    
        .. math::
            \text{UCB}(x) = \mu(x) + \kappa \sigma(x)

        where :math:`\Phi` is the CDF and :math:`\phi` the PDF of the normal
        distribution.

        Parameters
        ----------
        x : np.ndarray
            Parameters to evaluate the function at.

        gp : sklearn.gaussian_process.GaussianProcessRegressor
            A gaussian process regressor modelling the target function based on
            previous observations.

        y_max : number
            Highest found value of the target function.

        kappa : float, positive
            Governs the exploration/exploitation tradeoff. Lower prefers
            exploitation, higher prefers exploration.


        Returns
        -------
        Values of the acquisition function
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def ei(x, gp, y_max, xi):
        r"""Calculate Expected Improvement acqusition function.

        Similar to Probability of Improvement (`UtilityFunction.poi`), but also considers the
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
        x : np.ndarray
            Parameters to evaluate the function at.

        gp : sklearn.gaussian_process.GaussianProcessRegressor
            A gaussian process regressor modelling the target function based on
            previous observations.
        
        y_max : number
            Highest found value of the target function.
            
        xi : float, positive
            Governs the exploration/exploitation tradeoff. Lower prefers
            exploitation, higher prefers exploration.


        Returns
        -------
        Values of the acquisition function
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        a = mean - y_max - xi
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def poi(x, gp, y_max, xi):
        r"""Calculate Probability of Improvement acqusition function.

        Calculated as
    
        .. math:: \text{POI}(x) = \Phi\left( \frac{\mu(x)-y_{\text{max}} -  \xi }{\sigma(x)} \right)

        where :math:`\Phi` is the CDF of the normal distribution.

        Parameters
        ----------
        x : np.ndarray
            Parameters to evaluate the function at.
        gp : sklearn.gaussian_process.GaussianProcessRegressor
            A gaussian process regressor modelling the target function based on
            previous observations.
        
        y_max : number
            Highest found value of the target function.
            
        xi : float, positive
            Governs the exploration/exploitation tradeoff. Lower prefers
            exploitation, higher prefers exploration.


        Returns
        -------
        Values of the acquisition function
        
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi)/std
        return norm.cdf(z)


class NotUniqueError(Exception):
    """A point is non-unique."""


def load_logs(optimizer, logs):
    """Load previous ...

    Parameters
    ----------
    optimizer : BayesianOptimizer
        Optimizer the register the previous observations with.

    logs : str or bytes or os.PathLike
        File to load the logs from.

    Returns
    -------
    The optimizer with the state loaded.
    
    """
    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"],
                        constraint_value=(
                            iteration["constraint"]
                            if optimizer.is_constrained else None
                        )
                    )
                except NotUniqueError:
                    continue

    return optimizer


def ensure_rng(random_state=None):
    """Create a random number generator based on an optional seed.

    Parameters
    ----------
    random_state : np.random.RandomState or int or None, default=None
        Random state to use. if `None`, will create an unseeded random state.
        If `int`, creates a state using the argument as seed. If a
        `np.random.RandomState` simply returns the argument.

    Returns
    -------
    np.random.RandomState
    
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state

