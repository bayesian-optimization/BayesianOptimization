from __future__ import print_function
from __future__ import division

import numpy as np
from datetime import datetime
from scipy.stats import norm
from math import exp, fabs, log, pi


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

    def utility(self, x, gp, ymax):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, ymax)
        if self.kind == 'poi':
            return self._ucb(x, gp, ymax)

    @staticmethod
    def _ucb(x, gp, kappa):
        mean, var = gp.predict(x, eval_MSE=True)
        return mean + kappa * np.sqrt(var)

    @staticmethod
    def _ei(x, gp, ymax):
        mean, var = gp.predict(x, eval_MSE=True)
        if var == 0:
            return 0
        else:
            Z = (mean - ymax)/np.sqrt(var)
            return (mean - ymax) * norm.cdf(Z) + np.sqrt(var) * norm.pdf(Z)

    @staticmethod
    def _poi(x, gp, ymax):
        mean, var = gp.predict(x, eval_MSE=True)
        if var == 0:
            return 1
        else:
            Z = (mean - ymax)/np.sqrt(var)
            return norm.cdf(Z)


class PrintInfo(object):
    """
    A class to take care of the verbosity of the other classes.
    """

    def __init__(self, level=0):

        self.lvl = level
        self.timer = 0

    def print_info(self, op_start, i, x_max, ymax, xtrain, ytrain, keys):

        if self.lvl:
            np.set_printoptions(precision=4, suppress=True)
            print('Iteration: %3i | Last sampled value: %11f' % ((i+1), ytrain[-1]), '| with parameters: ', dict(zip(keys, xtrain[-1])))
            print('               | Current maximum: %14f | with parameters: ' % ymax, dict(zip(keys, xtrain[np.argmax(ytrain)])))
            
            minutes, seconds = divmod((datetime.now() - op_start).total_seconds(), 60)
            print('               | Time taken: %i minutes and %s seconds' % (minutes, seconds))
            print('')

        else:
            pass

    def print_log(self, op_start, i, x_max, xmins, min_max_ratio, ymax, xtrain, ytrain, keys):

        def return_log(x):
            return xmins * (10 ** (x * min_max_ratio))

        dict_len = len(keys)

        if self.lvl:
                
            np.set_printoptions(precision=4, suppress=True)
            print('Iteration: %3i | Last sampled value: %8f' % ((i+1), ytrain[-1]), '| with parameters: ',  dict(zip(keys, return_log(xtrain[-1])) ))
            print('               | Current maximum: %11f | with parameters: ' % ymax, dict(zip(keys, return_log(xtrain[np.argmax(ytrain)]))))

            minutes, seconds = divmod((datetime.now() - op_start).total_seconds(), 60)
            print('               | Time taken: %i minutes and %s seconds' % (minutes, seconds))
            print('')

        else:
            pass
