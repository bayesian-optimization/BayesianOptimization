import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm


class ConstraintModel():
    """
    This class takes the function to optimize as well as the parameters bounds
    in order to find which values for the parameters yield the maximum value
    using bayesian optimization.

    Parameters
    ----------
    func: function
        Constraint function. If multiple constraints are handled, this should
        return a numpy.ndarray of appropriate size.

    limits: numeric or numpy.ndarray
        Upper limit(s) for the constraints. The return value of `func` should
        have exactly this shape.

    random_state: int or numpy.random.RandomState, optional(default=None)
        If the value is an integer, it is used as the seed for creating a
        numpy.random.RandomState. Otherwise the random state provided is used.
        When set to None, an unseeded random state is generated.

    Note
    ----
    In case of multiple constraints, this model assumes conditional
    independence. This means that for each constraint, the probability of
    fulfillment is the cdf of a univariate Gaussian. The overall probability
    is a simply the product of the individual probabilities.
    """

    def __init__(self, func, limits, random_state=None):
        self.func = func

        if isinstance(limits, float):
            self._limits = np.array([limits])
        else:
            self._limits = limits

        basis = lambda: GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state,
        )
        self._model = [basis() for _ in range(len(self._limits))]

    @property
    def limits(self):
        return self._limits

    def eval(self, **kwargs):
        """
        Evaluates the constraint function.
        """
        try:
            return self.func(**kwargs)
        except TypeError as e:
            msg = (
                "Encountered TypeError when evaluating constraint " +
                "function. This could be because your constraint function " +
                "doesn't use the same keyword arguments as the target " +
                f"function. Original error message:\n\n{e}"
                )
            raise TypeError(msg)

    def fit(self, X, Y):
        """
        Fits internal GaussianProcessRegressor's to the data.
        """
        if len(self._model) == 1:
            self._model[0].fit(X, Y)
        else:
            for i, gp in enumerate(self._model):
                gp.fit(X, Y[:, i])

    def predict(self, X):
        """
        Returns the probability that the constraint is fulfilled at `X` based
        on the internal Gaussian Process Regressors.

        Note that this does not try to approximate the values of the constraint
        function, but probability that the constraint function is fulfilled.
        For the former, see `ConstraintModel.approx()`.
        """
        X_shape = X.shape
        X = X.reshape((-1, self._model[0].n_features_in_))
        if len(self._model) == 1:
            y_mean, y_std = self._model[0].predict(X, return_std=True)
            result = norm(loc=y_mean, scale=y_std).cdf(self._limits[0])
            return result.reshape(X_shape[:-1])
        else:
            result = np.ones(X.shape[0])
            for j, gp in enumerate(self._model):
                y_mean, y_std = gp.predict(X, return_std=True)
                result = result * norm(loc=y_mean, scale=y_std).cdf(
                    self._limits[j])
            return result.reshape(X_shape[:-1])

    def approx(self, X):
        """
        Returns the approximation of the constraint function using the internal
        Gaussian Process Regressors.
        """
        X_shape = X.shape
        X = X.reshape((-1, self._model[0].n_features_in_))
        if len(self._model) == 1:
            return self._model[0].predict(X).reshape(X_shape[:-1])
        else:
            result = np.column_stack([gp.predict(X) for gp in self._model])
            return result.reshape(X_shape[:-1] + (len(self._limits), ))

    def allowed(self, constraint_values):
        """
        Checks whether `constraint_values` are below the specified limits.
        """
        if self._limits.size == 1:
            return np.less_equal(constraint_values, self._limits)

        return np.all(constraint_values <= self._limits, axis=-1)
