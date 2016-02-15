from __future__ import print_function
from __future__ import division
import numpy as np
from datetime import datetime
from scipy.stats import norm


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind='ucb', kappa=1.96):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.kappa = kappa

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ui, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max)
        if self.kind == 'poi':
            return self._ucb(x, gp, y_max)

    @staticmethod
    def _ucb(x, gp, kappa):
        mean, var = gp.predict(x, eval_MSE=True)
        return mean + kappa * np.sqrt(var)

    @staticmethod
    def _ei(x, gp, y_max):
        mean, var = gp.predict(x, eval_MSE=True)
        if var == 0:
            return 0
        else:
            z = (mean - y_max)/np.sqrt(var)
            return (mean - y_max) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max):
        mean, var = gp.predict(x, eval_MSE=True)
        if var == 0:
            return 1
        else:
            z = (mean - y_max)/np.sqrt(var)
            return norm.cdf(z)


def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


def print_info(op_start, i, y_max, x_train, y_train, keys):

    np.set_printoptions(precision=4, suppress=True)
    print('Iteration: %3i | Last sampled value: %11f' % ((i+1), y_train[-1]),
          '| with parameters: ', dict(zip(keys, x_train[-1])))
    print('               | Current maximum: %14f | with parameters: ' % y_max,
          dict(zip(keys, x_train[np.argmax(y_train)])))

    minutes, seconds = divmod((datetime.now() - op_start).total_seconds(), 60)
    print('               | Time taken: %i minutes and %s seconds' % (minutes, seconds))
    print('')
