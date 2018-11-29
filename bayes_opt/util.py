import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def generate_trials(n_events, bounds, btypes, random_state):
    """A function to generate set of events under several constrains

    Parameters
    ----------
    :param n_events:
    The number of events to generate

    :param bounds:
    The variables bounds to limit the search of the acq max.

    :param btypes:
    The types of the variables.

    :param random_state:
    Instance of np.RandomState random number generator
    """
    x_trials = np.empty((n_events, bounds.shape[0]))
    if btypes is None:
        x_trials = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                        size=(n_events, bounds.shape[0]))
    else:
        for col, name in enumerate(bounds):
            # print(col, name)
            lower, upper = name
            if btypes[col] != int:
                x_trials[:, col] = random_state.uniform(lower, upper, size=n_events)
            if btypes[col] == int:
                x_trials[:, col] = random_state.randint(int(lower), int(upper), size=n_events)
    return x_trials


def acq_max(ac, gp, y_max, bounds, random_state, btypes=None, n_warmup=100000, n_iter=250):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.

    :param btypes:
        The types of the variables.

    :param random_state:
        instance of np.RandomState random number generator

    :param n_warmup:
        number of times to randomly sample the aquisition function

    :param n_iter:
        number of times to run scipy.minimize

    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """
    # Warm up with random points
    x_tries = generate_trials(n_warmup, bounds, btypes, random_state)
    ys = ac(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more throughly
    x_seeds = generate_trials(n_iter, bounds, btypes, random_state)
    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        ac_op = lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max)
        res = minimize(ac_op,
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        # If integer in list of bounds
        # search minimum between surroundings integers of the detected extremal point
        if btypes is not None:
            if int in btypes:
                x_inf = res.x.copy()
                x_sup = res.x.copy()
                for i, (val, t) in enumerate(zip(res.x, btypes)):
                    x_inf[i] = t(val)
                    x_sup[i] = t(val + 1) if t == int else t(val)
                # Store it if better than previous minimum(maximum).
                x_ext = [x_inf, x_sup]
                if max_acq is None or -res.fun[0] >= max_acq:
                    max_acq = -1*np.minimum(ac_op(x_inf), ac_op(x_sup))
                    x_argmax = np.argmin((ac_op(x_inf), ac_op(x_sup)))
                    x_max = x_ext[x_argmax]
            else:
                # If only float in bounds
                # store it if better than previous minimum(maximum).
                if max_acq is None or -res.fun[0] >= max_acq:
                    x_max = res.x
                    max_acq = -res.fun[0]
        else:
            if max_acq is None or -res.fun[0] >= max_acq:
                x_max = res.x
                max_acq = -res.fun[0]

        # Store it if better than previous minimum(maximum).
        # if max_acq is None or -res.fun[0] >= max_acq:
        #     x_max = res.x
        #     max_acq = -res.fun[0]

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.kappa = kappa

        self.xi = xi

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi)/std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi)/std
        return norm.cdf(z)


def load_logs(optimizer, logs):
    """Load previous ...

    """
    import json

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
                    )
                except KeyError:
                    pass

    return optimizer


def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


class Colours:
    """Print in nice colours."""

    BLUE = '\033[94m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    END = '\033[0m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'
    RED = '\033[91m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[93m'

    @classmethod
    def _wrap_colour(cls, s, colour):
        return colour + s + cls.END

    @classmethod
    def black(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.END)

    @classmethod
    def blue(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.BLUE)

    @classmethod
    def bold(cls, s):
        """Wrap text in bold."""
        return cls._wrap_colour(s, cls.BOLD)

    @classmethod
    def cyan(cls, s):
        """Wrap text in cyan."""
        return cls._wrap_colour(s, cls.CYAN)

    @classmethod
    def darkcyan(cls, s):
        """Wrap text in darkcyan."""
        return cls._wrap_colour(s, cls.DARKCYAN)

    @classmethod
    def green(cls, s):
        """Wrap text in green."""
        return cls._wrap_colour(s, cls.GREEN)

    @classmethod
    def purple(cls, s):
        """Wrap text in purple."""
        return cls._wrap_colour(s, cls.PURPLE)

    @classmethod
    def red(cls, s):
        """Wrap text in red."""
        return cls._wrap_colour(s, cls.RED)

    @classmethod
    def underline(cls, s):
        """Wrap text in underline."""
        return cls._wrap_colour(s, cls.UNDERLINE)

    @classmethod
    def yellow(cls, s):
        """Wrap text in yellow."""
        return cls._wrap_colour(s, cls.YELLOW)


# def unique_rows(a):
#     """
#     A function to trim repeated rows that may appear when optimizing.
#     This is necessary to avoid the sklearn GP object from breaking

#     :param a: array to trim repeated rows from

#     :return: mask of unique rows
#     """
#     if a.size == 0:
#         return np.empty((0,))

#     # Sort array and kep track of where things should go back to
#     order = np.lexsort(a.T)
#     reorder = np.argsort(order)

#     a = a[order]
#     diff = np.diff(a, axis=0)
#     ui = np.ones(len(a), 'bool')
#     ui[1:] = (diff != 0).any(axis=1)

#     return ui[reorder]
