import warnings
import numpy as np
from numpy.random import RandomState
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import softmax
from .target_space import TargetSpace
from .constraint import ConstraintModel
from sklearn.gaussian_process import GaussianProcessRegressor
from typing import Callable, List, Union, Tuple
from copy import deepcopy
from numbers import Number

class ConstraintNotSupportedError(Exception):
    pass


class AcquisitionFunction():
    def __init__(self, random_state=None):
        if random_state is not None:
            if isinstance(random_state, RandomState):
                self.random_state = random_state
            else:
                self.random_state = RandomState(random_state)
        else:
            self.random_state = RandomState()
        self.i = 0

    def _fit_gp(self, gp: GaussianProcessRegressor, target_space: TargetSpace) -> None:
        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(target_space.params, target_space.target)
            if target_space.constraint is not None:
                target_space.constraint.fit(target_space.params, target_space._constraint_values)

    def suggest(self, gp: GaussianProcessRegressor, target_space: TargetSpace, n_random=10_000, n_l_bfgs_b=10, fit_gp:bool=True):
        if len(target_space) == 0:
            raise ValueError("Cannot suggest a point without previous samples. Use target_space.random_sample() to generate a point.")
        self.i += 1

    def _get_acq(self, base_acq: Callable, dim: int, gp: GaussianProcessRegressor, constraint: Union[ConstraintModel, None] = None) -> Callable:
        if constraint is not None:
            def acq(x):
                x = x.reshape(-1, dim)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mean, std = gp.predict(x, return_std=True)
                    p_constraints = constraint.predict(x)
                return -1 * base_acq(mean, std) * p_constraints
        else:
            def acq(x):
                x = x.reshape(-1, dim)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mean, std = gp.predict(x, return_std=True)
                return -1 * base_acq(mean, std)
        return acq

    def _acq_min(self, acq: Callable, bounds: np.ndarray, n_random=10_000, n_l_bfgs_b=10) -> np.ndarray:
        x_min_r, min_acq_r = self._random_sample_minimize(acq, bounds, n_random=n_random)
        x_min_l, min_acq_l = self._l_bfgs_b_minimize(acq, bounds, n_x_seeds=n_l_bfgs_b)
        if min_acq_r < min_acq_l:
            return x_min_r
        else:
            return x_min_l

    def _random_sample_minimize(self, acq: Callable, bounds: np.ndarray, n_random: int) -> Tuple[np.ndarray, float]:
        # Warm up with random points
        x_tries = self.random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_random, bounds.shape[0]))
        ys = acq(x_tries)
        x_min = x_tries[ys.argmin()]
        min_acq = ys.min()
        return x_min, min_acq
    
    def _l_bfgs_b_minimize(self, acq: Callable, bounds: np.ndarray, n_x_seeds:int=10) -> Tuple[np.ndarray, float]:
        x_seeds = self.random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_x_seeds, bounds.shape[0]))
        
        max_acq = None
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res = minimize(acq,
                        x_try,
                        bounds=bounds,
                        method="L-BFGS-B")

            # See if success
            if not res.success:
                continue

            # Store it if better than previous minimum(maximum).
            if max_acq is None or np.squeeze(res.fun) >= max_acq:
                x_max = res.x
                max_acq = np.squeeze(res.fun)

        # Clip output to make sure it lies within the bounds. Due to floating
        # point technicalities this is not always the case.
        return np.clip(x_max, bounds[:, 0], bounds[:, 1]), max_acq


class UpperConfidenceBound(AcquisitionFunction):
    def __init__(self, kappa=2.576, exploration_decay=None, exploration_decay_delay=None, random_state=None) -> None:
        super().__init__(random_state=random_state)
        self.kappa = kappa
        self.exploration_decay = exploration_decay
        self.exploration_decay_delay = exploration_decay_delay

    def suggest(self, gp: GaussianProcessRegressor, target_space: TargetSpace, n_random=10_000, n_l_bfgs_b=10, fit_gp:bool=True) -> np.ndarray:
        super().suggest(gp=gp, target_space=target_space, n_random=n_random, n_l_bfgs_b=n_l_bfgs_b, fit_gp=fit_gp)
        if target_space.constraint is not None:
            msg = (
                f"Received constraints, but acquisition function {type(self)} "
                + "does not support constrained optimization."
            )
            raise ConstraintNotSupportedError(msg)
        if fit_gp:
            self._fit_gp(gp=gp, target_space=target_space)

        def base_acq(mean, std):
            return mean + self.kappa * std
    
        acq = self._get_acq(base_acq, dim=target_space.bounds.shape[0], gp=gp)
        x_max = self._acq_min(acq, target_space.bounds, n_random=n_random, n_l_bfgs_b=n_l_bfgs_b)
        self.update_params()
        return x_max
    
    def update_params(self) -> None:
        if self.exploration_decay is not None:
            if self.exploration_decay_delay is None or self.exploration_decay_delay <= self.i:
                self.kappa = self.kappa*self.exploration_decay



class ProbabilityOfImprovement(AcquisitionFunction):
    def __init__(self, xi, exploration_decay=None, exploration_decay_delay=None, random_state=None) -> None:
        super().__init__(random_state=random_state)
        self.xi = xi
        self.exploration_decay = exploration_decay
        self.exploration_decay_delay = exploration_decay_delay

    def suggest(self, gp: GaussianProcessRegressor, target_space: TargetSpace, n_random=10_000, n_l_bfgs_b=10, fit_gp:bool=True) -> np.ndarray:
        super().suggest(gp=gp, target_space=target_space, n_random=n_random, n_l_bfgs_b=n_l_bfgs_b, fit_gp=fit_gp)
        if fit_gp:
            self._fit_gp(gp=gp, target_space=target_space)
        y_max = target_space.max()['target']

        def base_acq(mean, std):
            z = (mean - y_max - self.xi)/std
            return norm.cdf(z)

        acq = self._get_acq(base_acq, dim=target_space.bounds.shape[0], gp=gp, constraint=target_space.constraint)
        x_max = self._acq_min(acq, target_space.bounds, n_random=n_random, n_l_bfgs_b=n_l_bfgs_b)
        self.update_params()
        return x_max

    def update_params(self) -> None:
        if self.exploration_decay is not None:
            if self.exploration_decay_delay is None or self.exploration_decay_delay <= self.i:
                self.xi = self.xi*self.exploration_decay


class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, xi, exploration_decay=None, exploration_decay_delay=None, random_state=None) -> None:
        super().__init__(random_state=random_state)
        self.xi = xi
        self.exploration_decay = exploration_decay
        self.exploration_decay_delay = exploration_decay_delay
    
    def suggest(self, gp: GaussianProcessRegressor, target_space: TargetSpace, n_random=10_000, n_l_bfgs_b=10, fit_gp:bool=True) -> np.ndarray:
        super().suggest(gp=gp, target_space=target_space, n_random=n_random, n_l_bfgs_b=n_l_bfgs_b, fit_gp=fit_gp)
        if fit_gp:
            self._fit_gp(gp=gp, target_space=target_space)
        y_max = target_space.max()['target']

        def base_acq(mean, std):
            a = (mean - y_max - self.xi)
            z = a / std
            return a * norm.cdf(z) + std * norm.pdf(z)

        acq = self._get_acq(base_acq, dim=target_space.bounds.shape[0], gp=gp, constraint=target_space.constraint)
        x_max = self._acq_min(acq, target_space.bounds, n_random=n_random, n_l_bfgs_b=n_l_bfgs_b)
        self.update_params()
        return x_max

    def update_params(self) -> None:
        if self.exploration_decay is not None:
            if self.exploration_decay_delay is None or self.exploration_decay_delay <= self.i:
                self.xi = self.xi*self.exploration_decay


class ConstantLiar(AcquisitionFunction):
    def __init__(self, base_acquisition: AcquisitionFunction, strategy='min', random_state=None, atol=1e-5, rtol=1e-8) -> None:
        super().__init__(random_state)
        self.base_acquisition = base_acquisition
        self.dummies = []
        if not isinstance(strategy, Number) and not strategy in ['min', 'mean', 'max']:
            raise ValueError(f"Received invalid argument {strategy} for strategy.")
        self.strategy = strategy
        self.atol = atol
        self.rtol = rtol
    
    def _copy_target_space(self, target_space: TargetSpace) -> TargetSpace:
        keys = target_space.keys
        pbounds = {key: bound for key, bound in zip(keys, target_space.bounds)}
        target_space_copy = TargetSpace(
            None,
            pbounds=pbounds,
            constraint=target_space.constraint,
            allow_duplicate_points=target_space._allow_duplicate_points
        )
        target_space_copy._params = deepcopy(target_space._params)
        target_space_copy._target = deepcopy(target_space._target)

        return target_space_copy

    def _remove_expired_dummies(self, target_space: TargetSpace) -> None:
        # Could also be a one-liner list-comprehension, but this seems more understandable
        dummies = []
        for dummy in self.dummies:
            close = np.isclose(dummy, target_space.params, rtol=self.rtol, atol=self.atol)
            if not close.all(axis=1).any():
                dummies.append(dummy)
        self.dummies = dummies
        
    def suggest(self, gp: GaussianProcessRegressor, target_space: TargetSpace, n_random=10_000, n_l_bfgs_b=10, fit_gp:bool=True) -> np.ndarray:
        super().suggest(gp=gp, target_space=target_space, n_random=n_random, n_l_bfgs_b=n_l_bfgs_b, fit_gp=fit_gp)
        if target_space.constraint is not None:
            msg = (
                f"Received constraints, but acquisition function {type(self)} "
                + "does not support constrained optimization."
            )
            raise ConstraintNotSupportedError(msg)
    
        self._remove_expired_dummies(target_space)
        dummy_target_space = self._copy_target_space(target_space)
    
        if isinstance(self.strategy, Number):
            dummy_target = self.strategy
        elif self.strategy == 'min':
            dummy_target = target_space.target.min()
        elif self.strategy == 'mean':
            dummy_target = target_space.target.mean()
        else:
            assert self.strategy == 'max'
            dummy_target = target_space.target.max()

        for dummy in self.dummies:
            dummy_target_space.register(dummy, dummy_target)

        self._fit_gp(gp=gp, target_space=dummy_target_space)
        x_max = self.base_acquisition.suggest(gp, dummy_target_space, n_random=n_random, n_l_bfgs_b=n_l_bfgs_b, fit_gp=True)
        self.dummies.append(x_max)

        return x_max

class KrigingBeliever(AcquisitionFunction):
    def __init__(self, base_acquisition: AcquisitionFunction, random_state=None, atol=1e-5, rtol=1e-8) -> None:
        super().__init__(random_state)
        self.base_acquisition = base_acquisition
        self.dummies = []
        self.dummy_targets = []
        self.dummy_constraints = []
        self.atol = atol
        self.rtol = rtol
    
    def _copy_target_space(self, target_space: TargetSpace) -> TargetSpace:
        keys = target_space.keys
        pbounds = {key: bound for key, bound in zip(keys, target_space.bounds)}
        target_space_copy = TargetSpace(
            None,
            pbounds=pbounds,
            constraint=target_space.constraint,
            allow_duplicate_points=target_space._allow_duplicate_points
        )
        target_space_copy._params = deepcopy(target_space._params)
        target_space_copy._target = deepcopy(target_space._target)
        if target_space.constraint is not None:
            target_space_copy._constraint_values = deepcopy(target_space._constraint_values)
        return target_space_copy

    def _ensure_dummies_match(self) -> None:
        if self.dummy_constraints:
            if len(self.dummy_constraints) != len(self.dummy_targets):
                msg = (
                    "Number of dummy constraints " +
                    f"{len(self.dummy_constraints)} doesn't match number of " +
                    f" dummy targets {len(self.dummy_targets)}. This can " +
                    "happen if constrained and unconstrained optimization is "+
                    "mixed."
                )
                raise ValueError(msg)

    def _remove_expired_dummies(self, target_space: TargetSpace) -> None:
        self._ensure_dummies_match()
        dummies = []
        dummy_constraints = []
        for i, dummy in enumerate(self.dummies):
            close = np.isclose(dummy, target_space.params, rtol=self.rtol, atol=self.atol)
            if not close.all(axis=1).any():
                dummies.append(dummy)
                if target_space.constraint is not None:
                    dummy_constraints.append(self.dummy_constraints[i])
        self.dummies = dummies
        self.dummy_constraints = dummy_constraints
        
    def suggest(self, gp: GaussianProcessRegressor, target_space: TargetSpace, n_random=10_000, n_l_bfgs_b=10, fit_gp:bool=True) -> np.ndarray:
        super().suggest(gp=gp, target_space=target_space, n_random=n_random, n_l_bfgs_b=n_l_bfgs_b, fit_gp=fit_gp)
        self._remove_expired_dummies(target_space)
        dummy_target_space = self._copy_target_space(target_space)

        for idx, dummy in enumerate(self.dummies):
            if dummy_target_space.constraint is not None:
                dummy_target_space.register(dummy, self.dummy_targets[idx].squeeze(), self.dummy_constraints[idx].squeeze())
            else:
                dummy_target_space.register(dummy, self.dummy_targets[idx].squeeze())
    
        x_max = self.base_acquisition.suggest(gp, dummy_target_space, n_random=n_random, n_l_bfgs_b=n_l_bfgs_b, fit_gp=True)

        self._fit_gp(gp=gp, target_space=dummy_target_space)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target_pred = gp.predict(x_max.reshape(1, -1))
        self.dummies.append(x_max)
        self.dummy_targets.append(target_pred)

        if dummy_target_space.constraint is not None:
            constraint_pred = dummy_target_space.constraint.approx(x_max)
            self.dummy_constraints.append(constraint_pred)

        return x_max


class GPHedge(AcquisitionFunction):
    def __init__(self, base_acquisitions: List[AcquisitionFunction], random_state=None) -> None:
        super().__init__(random_state)
        self.base_acquisitions = base_acquisitions
        self.n_acq = len(self.base_acquisitions)
        self.gains = np.zeros(self.n_acq)
        self.previous_candidates = None

    def _sample_idx_from_softmax_gains(self) -> int:
        cumsum_softmax_g = np.cumsum(softmax(self.gains))
        r = self.random_state.rand()
        idx = np.argmax(r <= cumsum_softmax_g) # Returns the first True value
        return idx

    def _update_gains(self, gp: GaussianProcessRegressor) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rewards = gp.predict(self.previous_candidates)
        self.gains += rewards

    def suggest(self, gp: GaussianProcessRegressor, target_space: TargetSpace, n_random=10_000, n_l_bfgs_b=10, fit_gp:bool=True) -> np.ndarray:
        super().suggest(gp=gp, target_space=target_space, n_random=n_random, n_l_bfgs_b=n_l_bfgs_b, fit_gp=fit_gp)
        self._fit_gp(gp=gp, target_space=target_space)

        if self.previous_candidates is not None:
            self._update_gains(gp)
            self.previous_candidates = None

        x_max = []
        for base_acq in self.base_acquisitions:
            x_max.append(
                base_acq.suggest(
                    gp=gp,
                    target_space=target_space,
                    n_random=n_random//self.n_acq,
                    n_l_bfgs_b=n_l_bfgs_b//self.n_acq,
                    fit_gp=False
                )
            )
        idx = self._sample_idx_from_softmax_gains()
        self.previous_candidates = np.array(x_max)
        return x_max[idx]
