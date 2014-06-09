'''
________ Bayesian optimization --- GP module ________

Issues:

See papers: http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
            http://arxiv.org/pdf/1012.2599v1.pdf
            http://www.gaussianprocess.org/gpml/

for references.

Fernando Nogueira
'''



import numpy
from datetime import datetime

from scipy.optimize import minimize

from math import exp, fabs, sqrt, log, pi
from ..support.objects import covariance, sample_covariance, kernels, acquisition, print_info
#from help_functions import covariance, sample_covariance, kernels, acquisition, print_info

# Python 2.7 users.
# from __future__ import print_function
# from __future__ import division


################################################################################
################################____GP_Class____################################
################################################################################

class GP:
    '''
    ________ Gaussian Process Class ________

    Parameters
    ----------

    kernel : defaults to 'squared_exp', can accept a user defined one. This is the
    similarity function the gaussian process builds the covariance matrix from.

    theta : Kernel parameter related to how strongly related the data points are,
    see the kernel class.

    l : Another kernel parameter related to how fast the correlation between points
    decay with increasing distance, see kernel class.

    noise : defaults to 1e-6, introduces diagonal noise to the covariance matrix,
    useful both to model noisy models as well as for numerical stability.


    Methods
    -------

    set_kernel : allows to user to change the kernel of an existing gp object.

    log_like : return the loglikelihood of the fitted model.

    fit : Fit the gaussian process to the given data.

    best_fit : Fit the gaussian process and maximizes the log likelihood as function
    of the kernel parameters.

    predict : Make a prediction based on the fitted model and returns the mean and
    covariance matrix for the given data.

    fast_predict : Make a prediction based on the fitted model and returns the mean
    and only the diagonal elements of the covariance matrix for the given data.

    sample_predict : Make a prediction for a single data point, return the predicted
    mean and variance of the sample. 

    '''

    def __init__(self, noise = 1e-6, kernel = 'squared_exp', theta = 2, l = 1):
        '''Initializes the GP object with default parameters.

           Member variables
           ----------------

           self.kernel : Stores the kernel of choice as a member variable.

           self.fit_flag : A member variable to keep track of whether the the fit
           methods has already being called.

           self.L : Stores the cholesky decomposition of the train covariance matrix.

           self.a : Stores a vector used for solving the linear system in the gp.

           self.ll : Stores the log likelihood of the fitted model.

           self.train : Stores the train data, necessary to make predictions, without
           the need to ask for train data again.

           self.noise : Member variable to store the noise value.


        '''
        
        # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
        if noise == 0:
            print('Non zero noise helps with numerical stability and it is strongly advised.')
        if noise < 0:
            raise RuntimeError('Negative noise was entered.')       
        self.noise = noise


        # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
        kn = kernels(theta, l)
        kernel_types = {'squared_exp' : kn.squared_exp, 'ARD_matern' : kn.ARD_matern, 'trivial' : kn.trivial}

        try:
            self.kernel = kernel_types[kernel]
            self.kernel_name = kernel
        except KeyError:
            print('Using a custom kernel.')
            self.kernel = kernel


        # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
        ##Gp fit parameters
        self.fit_flag = False
        self.L = 0 # Stores the cholesky decomposition of the kernel matrix
        self.a = 0 # Vector used in the fitting algorith
        self.ll = 0 # Stores the log likelihood of the fitted model
        self.train = 0 # Stores the x train data.
        
    def __str__(self):
        text = 'This is an object that performs non-linear interpolation using gaussian processes.\n'
        if self.fit_flag == False:
            text2 = 'Its parameters have not been set yet.'
        else:
            text2 = 'Its parameters have been set and the object is ready to go!'

        return text + text2

    def set_kernel(self, new_kernel = 'ARD_matern', theta = 1, l = 1):
        ''' Set a new kernel for the gaussian process.


            Parameters
            ----------
            xtrain : nd array with predictor values.
            
            ytrain : 1d array with target values.
  
            Returns
            -------
            Nothing.
        '''
        
        kn = kernels(theta, l)
        kernel_types = {'squared_exp' : kn.squared_exp, 'ARD_matern' : kn.ARD_matern, 'trivial' : kn.trivial}

        try:
            self.kernel = kernel_types[new_kernel]
        except KeyError:
            print('Using a custom kernel.')
            self.kernel = new_kernel
            

    def log_like(self):
        ''' Methods that return the log likelihood of the gaussian process.


            Parameters
            ----------
            No parameters.


            Returns
            -------
            The log likelihood of the currently fitted GP model.
        '''
        
        if self.fit_flag == False:
            raise RuntimeError('You have to fit the GP model first.')

        return self.ll



    # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
    # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
    def fit(self, xtrain, ytrain):
        ''' Method responsible for fitting the gaussian process. It follows the pseudo-code in the GP book.


            Parameters
            ----------
            xtrain : nd array with predictor values.
            
            ytrain : 1d array with target values.
  
            Returns
            -------
            Nothing.
        '''
        
        self.train = xtrain
        self.L = numpy.linalg.cholesky(covariance(self.train, self.train, kernel = self.kernel) + self.noise * numpy.eye(len(self.train)))
        self.a = numpy.linalg.solve(self.L.T, numpy.linalg.solve(self.L, ytrain))
        self.ll = -0.5 * numpy.dot(ytrain.T, self.a) - numpy.sum(numpy.log(numpy.diagonal(self.L))) - 0.5 * len(self.train) * log(2 * pi)
        self.fit_flag = True


    def best_fit(self, xtrain, ytrain):
        ''' This method perform the GP fit but it maximizes the log likelihood by varying the parameters of the kernel.
            The function log_res defines the function to be optimized. In case the optimization fails regular fit is performed.


            Parameters
            ----------
            xtrain : nd array with predictor values.
            
            ytrain : 1d array with target values.
  
            Returns
            -------
            Nothing.
        '''

        def log_res(para):
            self.set_kernel(new_kernel = self.kernel_name, theta = para[0], l = para[1])
            self.fit(xtrain, ytrain)
            return -self.log_like()
        
        try:
            res = minimize(log_res, [1,1], bounds = [[0.01,5],[0.01,2]], method = 'L-BFGS-B')
            self.set_kernel(self.kernel_name, theta = res.x[0], l = res.x[1])
            self.fit(xtrain, ytrain)
        except:
            print('not using best_fit')
            self.fit(xtrain, ytrain)


    # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
    # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
    def predict(self, xtest):
        ''' Method responsible for making predictions based of the fitted model for multiple new data points.


            Parameters
            ----------
            xtest : nd array with predictor values.

  
            Returns
            -------
            1d-array with mean values and the full covariance matrix.
        '''
        
        if self.fit_flag == False:
            raise RuntimeError('You have to fit the GP model first.')

        Ks = covariance(self.train, xtest, kernel = self.kernel)
        Kss = covariance(xtest, xtest, kernel = self.kernel, fast = False)
        v = numpy.linalg.solve(self.L, Ks)

        mean = numpy.dot(Ks.T, self.a)
        var = Kss - numpy.dot(v.T, v)
        
        return mean, var

    def fast_predict(self, xtest):
        ''' Method responsible for making predictions based of the fitted model for multiple new data points.


            Parameters
            ----------
            xtest : nd array with predictor values.

  
            Returns
            -------
            1d-array with mean values and the diagonal of the covariance matrix.
        '''
        
        if self.fit_flag == False:
            raise RuntimeError('You have to fit the GP model first.')

        Ks = covariance(self.train, xtest, kernel = self.kernel)
        Kss = covariance(xtest, xtest, kernel = self.kernel, fast = True)
        v = numpy.linalg.solve(self.L, Ks)

        mean = numpy.dot(Ks.T, self.a)
        var = numpy.diagonal(Kss - numpy.dot(v.T, v))
        
        return mean, var

    def sample_predict(self, xtest):
        ''' Method responsible for making predictions based of the fitted model for a single new data point.


            Parameters
            ----------
            xtest : 1d array with predictor values.

  
            Returns
            -------
            The mean and covariance for the predicted point.
        '''
        
        if self.fit_flag == False:
            raise RuntimeError('You have to fit the GP model first.')

        Ks = sample_covariance(self.train, xtest, kernel = self.kernel)
        Kss = self.kernel(xtest, xtest)
        v = numpy.linalg.solve(self.L, Ks)

        mean = numpy.dot(Ks.T, self.a).reshape(1)
        var = Kss - numpy.dot(v.T, v)
        
        return mean[0], var
