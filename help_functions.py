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
#################################### Kernels ###################################
################################################################################

class kernels:
    '''Object with all kernels as methods'''
    def __init__(self, t = 1, l = 1):
        self.t = t
        self.l = l

    def squared_exp(self, x1, x2):
        '''
        This is the kernel function used to calculate the
        covariance matrix. Right now I am using the standard
        double exponential one. Later I want to include
        a few different ones such as the ARD Matern ´ 5/2 kernel.
        '''
        return (self.t**2) * exp(-numpy.dot(x1 - x2, x1 - x2) / (2 * self.l**2))


    def ARD_matern(self, x1, x2):
        '''
        This is the kernel function used to calculate the
        covariance matrix
        '''
        r2 = numpy.dot(x1 - x2, x1 - x2) / (self.l**2)
    
        return (self.t ** 2) * (1 + sqrt(5 * r2) + (5/3) * r2) * exp(-sqrt(5 * r2))

    def trivial(self, x1, x2):
        return self.t * numpy.dot(x1, x2)

################################################################################
################################ Covariance calc ###############################
################################################################################

def covariance(X1, X2, kernel, fast = False):
    '''
    This ugly functions builds the covariance matrix given a pair
    of matrices. It can be done much more elegantly,
    but this is the solution I came up with and for now it is staying
    like this.

    Note the shapes of the arrays X1 and X2 must be changed to matrix like!
    '''

    try:
        X1.shape[1]
    except:
        X1 = X1.reshape((1, len(X1)))

    try:
        X2.shape[1]
    except:
        X2 = X2.reshape((1, len(X2)))


    if fast:
        M = numpy.zeros(len(X1))                       
        for row in range(len(X2)):
            M[row] = kernel(X1[row, :], X2[row, :])
    else:
        M = numpy.zeros((len(X1), len(X2)))
        for row1 in range(len(X1)):
            for row2 in range(len(X2)):
                M[row1, row2] = kernel(X1[row1, :], X2[row2, :])

    return M

def sample_covariance(X1, sample, kernel):
    '''
    This ugly functions builds the covariance matrix given a pair
    of matrices. It can be done much more elegantly,
    but this is the solution I came up with and for now it is staying
    like this.

    Note the shapes of the arrays X1 and X2 must be changed to matrix like!
    '''
    M = numpy.zeros(len(X1))
    
    for row in range(len(X1)):
        M[row] = kernel(X1[row, :], sample)

    return M.reshape(len(X1))

################################################################################
############################# Aquisition Functions #############################
################################################################################

class acquisition:
    '''An object to compute the acquisition functions.'''


    def __init__(self, k = 1):
        '''If UCB is to be used, a constant kappa is needed.'''
        self.kappa = k

    def UCB(self, x, gp, ymax):
        mean, var = gp.sample_predict(x)
        return mean + self.kappa * var

    def EI(self, x, gp, ymax):
        mean, var = gp.sample_predict(x)
        if var == 0:
            return 0
        else:
            Z = (mean- ymax)/var
            return (mean - ymax) * norm.cdf(Z) + var * norm.pdf(Z)

    def PoI(self, x, gp, ymax):
        mean, var = gp.sample_predict(x)
        if var == 0:
            return 1
        else:
            Z = (mean- ymax)/var
            return norm.cdf(Z)

    # ------------------------------ // ------------------------------ #
    #Methods that I currently don't have a use for anymore.

    def full_PI(ymax, mean, var):
        '''
        Function to calculate the probability of improvement. In the current implementation
        it breaks down in the system has no noise (even though it shouldn't!). It can easily
        be fixed and I will do it later...
        '''
        mean = mean.reshape(len(mean))
        var = numpy.diagonal(var)

        gamma = (mean - ymax)/var
    
        return norm.cdf(gamma)


    def full_EI(ymax, mean, var, verbose = False):
        '''
        Function to calculate the expected improvement. Robust agains noiseless
        systems.
        '''
        if verbose:
            print('EI was called with ymax: %f' % ymax)

        ei = numpy.zeros(len(mean))

        mean = mean.reshape(len(mean))
        var = numpy.diagonal(var)

        Z = (mean[var > 0] - ymax)/var[var > 0]

        ei[var > 0] = (mean[var > 0] - ymax) * norm.cdf(Z) + var[var > 0] * norm.pdf(Z)

        return ei
