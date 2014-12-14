from __future__ import print_function
from __future__ import division

import numpy
from datetime import datetime
from scipy.stats import norm
from math import exp, fabs, sqrt, log, pi

################################################################################
################################################################################
################################ Help Functions ################################
################################################################################
################################################################################

################################################################################
############################# Aquisition Functions #############################
################################################################################

class AcquisitionFunction(object):
    '''An object to compute the acquisition functions.'''


    def __init__(self, k=1):
        '''If UCB is to be used, a constant kappa is needed.'''
        self.kappa = k

    # ------------------------------ // ------------------------------ #
    # Methods for single sample calculation.
    def UCB(self, x, gp, ymax):
        mean, var = gp.predict(x, eval_MSE=True)
        return mean + self.kappa * sqrt(var)

    def EI(self, x, gp, ymax):
        mean, var = gp.predict(x, eval_MSE=True)
        if var == 0:
            return 0
        else:
            Z = (mean - ymax)/sqrt(var)
            return (mean - ymax) * norm.cdf(Z) + sqrt(var) * norm.pdf(Z)

    def PoI(self, x, gp, ymax):
        mean, var = gp.predict(x, eval_MSE=True)
        if var == 0:
            return 1
        else:
            Z = (mean - ymax)/sqrt(var)
            return norm.cdf(Z)

    # ------------------------------ // ------------------------------ #
    # Methods for bulk calculation.
    def full_UCB(self, mean, var):
        mean = mean.reshape(len(mean))
        
        return (mean + self.kappa * numpy.sqrt(var)).reshape(len(mean))


    def full_EI(self, ymax, mean, var, verbose = False):
        '''
        Function to calculate the expected improvement. Robust agains noiseless
        systems.
        '''
        if verbose:
            print('EI was called with ymax: %f' % ymax)

        ei = numpy.zeros(len(mean))

        mean = mean.reshape(len(mean))
        var = numpy.sqrt(var)

        Z = (mean[var > 0] - ymax)/var[var > 0]

        ei[var > 0] = (mean[var > 0] - ymax) * norm.cdf(Z) + var[var > 0] * norm.pdf(Z)

        return ei

    def full_PoI(self, ymax, mean, var):
        '''
        Function to calculate the probability of improvement. In the current implementation
        it breaks down in the system has no noise (even though it shouldn't!). It can easily
        be fixed and I will do it later...
        '''
        mean = mean.reshape(len(mean))
        var = numpy.sqrt(var)

        gamma = (mean - ymax)/var
    
        return norm.cdf(gamma)


################################################################################
################################## Print Info ##################################
################################################################################


class PrintInfo(object):
    '''A class to take care of the verbosity of the other classes.'''
    '''Under construction!'''

    def __init__(self, level=0):

        self.lvl = level
        self.timer = 0


    def print_info(self, op_start, i, x_max, ymax, xtrain, ytrain, keys):

        if self.lvl:
            numpy.set_printoptions(precision=4, suppress=True)
            print('Iteration: %3i | Last sampled value: %11f' % ((i+1), ytrain[-1]), '| with parameters: ', dict(zip(keys, xtrain[-1])))
            print('               | Current maximum: %14f | with parameters: ' % ymax, dict(zip(keys, xtrain[numpy.argmax(ytrain)])))
            
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
                
            numpy.set_printoptions(precision=4, suppress=True)
            print('Iteration: %3i | Last sampled value: %8f' % ((i+1), ytrain[-1]), '| with parameters: ',  dict(zip(keys, return_log(xtrain[-1])) ))
            print('               | Current maximum: %11f | with parameters: ' % ymax, dict(zip(keys, return_log( xtrain[numpy.argmax(ytrain)]))))

            minutes, seconds = divmod((datetime.now() - op_start).total_seconds(), 60)
            print('               | Time taken: %i minutes and %s seconds' % (minutes, seconds))
            print('')

        else:
            pass
