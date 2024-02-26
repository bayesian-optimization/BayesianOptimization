"""Constraint handling."""
import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm


class ConstraintModel():
    """Model constraints using GP regressors.

    This class takes the function to optimize as well as the parameters bounds
    in order to find which values for the parameters yield the maximum value
    using bayesian optimization.

    Parameters
    ----------
    fun : None or Callable -> float or np.ndarray
        The constraint function. Should be float-valued or array-valued (if
        multiple constraints are present). Needs to take the same parameters
        as the optimization target with the same argument names.

    lb : float or np.ndarray
        The lower bound on the constraints. Should have the same
        dimensionality as the return value of the constraint function.

    ub : float or np.ndarray
        The upper bound on the constraints. Should have the same
        dimensionality as the return value of the constraint function.

    random_state : np.random.RandomState or int or None, default=None
        Random state to use.

    Notes
    -----
    In case of multiple constraints, this model assumes conditional
    independence. This means that for each constraint, the probability of
    fulfillment is the cdf of a univariate Gaussian. The overall probability
    is a simply the product of the individual probabilities.
    """

    def __init__(self, fun, lb, ub, random_state=None):
        self.fun = fun

        self._lb = np.atleast_1d(lb)
        self._ub = np.atleast_1d(ub)

        if np.any(self._lb >= self._ub):
            msg = "Lower bounds must be less than upper bounds."
            raise ValueError(msg)

        basis = lambda: GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state,
        )
        self._model = [basis() for _ in range(len(self._lb))]

    @property
    def lb(self):
        """Return lower bounds."""
        return self._lb

    @property
    def ub(self):
        """Return upper bounds."""
        return self._ub

    @property
    def model(self):
        """Return GP regressors of the constraint function."""
        return self._model

    def eval(self, **kwargs: dict):
        r"""Evaluate the constraint function.

        Parameters
        ----------
        \*\*kwargs :
            Function arguments to evaluate the constraint function on.


        Returns
        -------
        Value of the constraint function.

        Raises
        ------
        TypeError
            If the kwargs' keys don't match the function argument names.
        """
        try:
            return self.fun(**kwargs)
        except TypeError as e:
            msg = (
                "Encountered TypeError when evaluating constraint " +
                "function. This could be because your constraint function " +
                "doesn't use the same keyword arguments as the target " +
                f"function. Original error message:\n\n{e}"
                )
            e.args = (msg,)
            raise

    def fit(self, X, Y):
        """Fit internal GPRs to the data.

        Parameters
        ----------
        X :
            Parameters of the constraint function.
        Y :
            Values of the constraint function.


        Returns
        -------
        None
        """
        if len(self._model) == 1:
            self._model[0].fit(X, Y)
        else:
            for i, gp in enumerate(self._model):
                gp.fit(X, Y[:, i])

    def predict(self, X):
        r"""Calculate the probability that the constraint is fulfilled at `X`.

        Note that this does not try to approximate the values of the
        constraint function (for this, see `ConstraintModel.approx()`.), but
        probability that the constraint function is fulfilled. That is, this
        function calculates

        .. math::
            p = \text{Pr}\left\{c^{\text{low}} \leq \tilde{c}(x) \leq
                c^{\text{up}} \right\} = \int_{c^{\text{low}}}^{c^{\text{up}}}
                \mathcal{N}(c, \mu(x), \sigma^2(x)) \, dc.

        with :math:`\mu(x)`, :math:`\sigma^2(x)` the mean and variance at
        :math:`x` as given by the GP and :math:`c^{\text{low}}`,
        :math:`c^{\text{up}}` the lower and upper bounds of the constraint
        respectively.

        In case of multiple constraints, we assume conditional independence.
        This means we calculate the probability of constraint fulfilment
        individually, with the joint probability given as their product.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Parameters for which to predict the probability of constraint
            fulfilment.
            

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Probability of constraint fulfilment.

        """
        X_shape = X.shape
        X = X.reshape((-1, self._model[0].n_features_in_))
        if len(self._model) == 1:
            y_mean, y_std = self._model[0].predict(X, return_std=True)

            p_lower = (norm(loc=y_mean, scale=y_std).cdf(self._lb[0])
                            if self._lb[0] != -np.inf else np.array([0]))
            p_upper = (norm(loc=y_mean, scale=y_std).cdf(self._ub[0])
                            if self._lb[0] != np.inf else np.array([1]))
            result = p_upper - p_lower
            return result.reshape(X_shape[:-1])

        result = np.ones(X.shape[0])
        for j, gp in enumerate(self._model):
            y_mean, y_std = gp.predict(X, return_std=True)
            p_lower = (norm(loc=y_mean, scale=y_std).cdf(self._lb[j])
                        if self._lb[j] != -np.inf else np.array([0]))
            p_upper = (norm(loc=y_mean, scale=y_std).cdf(self._ub[j])
                        if self._lb[j] != np.inf else np.array([1]))
            result = result * (p_upper - p_lower)
        return result.reshape(X_shape[:-1])

    def approx(self, X):
        """
        Approximate the constraint function using the internal GPR model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Parameters for which to estimate the constraint function value.

        Returns
        -------
        np.ndarray of shape (n_samples, n_constraints)
            Constraint function value estimates.
        """
        X_shape = X.shape
        X = X.reshape((-1, self._model[0].n_features_in_))
        if len(self._model) == 1:
            return self._model[0].predict(X).reshape(X_shape[:-1])

        result = np.column_stack([gp.predict(X) for gp in self._model])
        return result.reshape(X_shape[:-1] + (len(self._lb), ))

    def allowed(self, constraint_values):
        """Check whether `constraint_values` fulfills the specified limits.

        Parameters
        ----------
        constraint_values : np.ndarray of shape (n_samples, n_constraints)
            The values of the constraint function.
            

        Returns
        -------
        np.ndarrray of shape (n_samples,)
            Specifying wheter the constraints are fulfilled.

        """
        if self._lb.size == 1:
            return (np.less_equal(self._lb, constraint_values)
                    & np.less_equal(constraint_values, self._ub))

        return (np.all(constraint_values <= self._ub, axis=-1)
                    & np.all(constraint_values >= self._lb, axis=-1))
