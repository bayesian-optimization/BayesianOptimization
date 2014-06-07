'''
________ Bayesian optimization ________

Issues: Pre initialization of Log_maximize is broken, must fix it.

See papers: http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
            http://arxiv.org/pdf/1012.2599v1.pdf
            http://www.gaussianprocess.org/gpml/

for references.
'''



import numpy
from datetime import datetime

from scipy.optimize import minimize

from math import exp, fabs, sqrt, log, pi
from help_functions import covariance, sample_covariance, kernels, acquisition, print_info

# Python 2.7 users.
# from __future__ import print_function
# from __future__ import division


################################################################################
################################____GP_Class____################################
################################################################################

class GP:
    '''
    ________ Gaussian Process Class ________

    Enter here the basic information regarding this class and how it works

    '''

    def __init__(self, noise = 1e-6, kernel = 'squared_exp', theta = 2, l = 1):
        '''Three different kernels'''
        
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
        self.train = 0 # Stores the x train data. Necessary to make predictions without the need to ask for train data everytime.

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
        ''' Methods responsible for fitting the gaussian process. It follows the pseudo-code in the GP book.


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
        if self.fit_flag == False:
            raise RuntimeError('You have to fit the GP model first.')

        Ks = covariance(self.train, xtest, kernel = self.kernel)
        Kss = covariance(xtest, xtest, kernel = self.kernel, fast = False)
        v = numpy.linalg.solve(self.L, Ks)

        mean = numpy.dot(Ks.T, self.a)
        var = Kss - numpy.dot(v.T, v)
        
        return mean, var

    def fast_predict(self, xtest):
        if self.fit_flag == False:
            raise RuntimeError('You have to fit the GP model first.')

        Ks = covariance(self.train, xtest, kernel = self.kernel)
        Kss = covariance(xtest, xtest, kernel = self.kernel, fast = True)
        v = numpy.linalg.solve(self.L, Ks)

        mean = numpy.dot(Ks.T, self.a)
        var = numpy.diagonal(Kss - numpy.dot(v.T, v))
        
        return mean, var

    def sample_predict(self, xtest):
        if self.fit_flag == False:
            raise RuntimeError('You have to fit the GP model first.')

        Ks = sample_covariance(self.train, xtest, kernel = self.kernel)
        Kss = self.kernel(xtest, xtest)
        v = numpy.linalg.solve(self.L, Ks)

        mean = numpy.dot(Ks.T, self.a).reshape(1)
        var = Kss - numpy.dot(v.T, v)
        
        return mean[0], var

    
    

################################################################################
##############################____Bayes_Class____###############################
################################################################################

class bayes_opt:
    '''
    ________ Optimization Class ________

    Enter here the basic information regarding this class and how it works

    '''

    def __init__(self, f, arr_tup, kernel = 'squared_exp', acq = 'ei', min_log = True):
        '''This is an object to find the global maximum of an unknown function via gaussian processes./n
           It takes a function of N variables and the lower and upper bounds for each variable as parameters.
           It also (will) accept the log_grid boolean appropriate for when the bounds are several orders of magnitude apart.'''
        
        for n, pair in enumerate(arr_tup):
            if pair[1] == pair[0]:
                raise RuntimeError('The upper and lower bound of parameter %i are the same, the upper bound must be greater than the lower bound.' % n)
            if pair[1] < pair[0]:
                raise RuntimeError('The upper bound of parameter %i is less than the lower bound, the upper bound must be greater than the lower bound.' % n)

        # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
        self.bounds = numpy.asarray(arr_tup)
        self.log_bounds = 0 * numpy.asarray(arr_tup)
        self.dim = len(arr_tup)

        if all(self.bounds[:, 0] > 0):
            ratio = self.bounds[:, [1]] / self.bounds[:, [0]]
            if any(ratio > 1000):
                print('The order of magnitude of some of your bounds is too different, you may benefit from using log_maximize.')



        # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
        self.f = f
        
        self.kernel = kernel
        self.k_theta = 2
        self.k_l = 1

        ac = acquisition()
        ac_types = {'ei' : ac.EI, 'pi' : ac.PoI, 'ucb' : ac.UCB}
        try:
            self.ac = ac_types[acq]
        except KeyError:
            print('Custom acquisition function being used.')
            self.ac = acq


        # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
        self.min_log = min_log
        self.user_init = False
        self.user_x = numpy.empty((1, len(arr_tup)))
        self.user_y = numpy.empty(1)


        
    # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
    # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
    def set_acquisition(self, acq = 'ucb', k = 1):
        '''A method to set the acquisition function to be used.'''
        
        ac = acquisition(k)
        ac_types = {'ei' : ac.EI, 'pi' : ac.PoI, 'ucb' : ac.UCB}
        try:
            self.ac = ac_types[acq]
        except KeyError:
            print('Custom acquisition function being used.')
            self.ac = acq

    def set_kernel(self, kernel = 'ARD_matern', theta = 1, l = 1):
        '''A method to set the kernel to be used.'''
        self.kernel = kernel
        self.k_theta = theta
        self.k_l = l

    # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
    def acq_max(self, gp, ymax, restarts, Bounds):

        x_max = Bounds[:, 0]
        ei_max = 0

        for i in range(restarts):
            #Sample some points at random.
            x_try = numpy.asarray([numpy.random.uniform(x[0], x[1], size = 1) for x in Bounds]).T
            
            #Find the minimum of minus que acquisition function
            res = minimize(lambda x: -self.ac(x, gp = gp, ymax = ymax), x_try, bounds = Bounds, method = 'L-BFGS-B')

            #Store it if better than previous minimum.
            if -res.fun >= ei_max:
                x_max = res.x
                ei_max = -res.fun

        return x_max


    # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
    # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
    def maximize(self, init_points = 3, restarts = 10, num_it = 15, verbose = 2, full_out = False):
        '''Main optimization method.'''

        # ------------------------------ // ------------------------------ // ------------------------------ #
        if not self.user_init:
            print('Optimization procedure is initializing at %i random points.' % init_points)
            xtrain = numpy.asarray([numpy.random.uniform(x[0], x[1], size = init_points) for x in self.bounds]).T
            ytrain = numpy.asarray([self.f(x) for x in xtrain])
            print('Optimization procedure is done initializing.')
        else:
            print('Optimization procedure is initializing at %i random points.' % init_points)
            new_xtrain = numpy.asarray([numpy.random.uniform(x[0], x[1], size = init_points) for x in self.bounds]).T
            
            xtrain = numpy.concatenate((self.user_x, new_xtrain), axis = 0)
            ytrain = numpy.concatenate((self.user_y, numpy.asarray([self.f(x) for x in new_xtrain])))
            print('Optimization procedure is done initializing.')



        pi = print_info(verbose)
        ymax = ytrain.max()

        # ------------------------------ // ------------------------------ // ------------------------------ #
        # Fitting the gaussian process.
        gp = GP(kernel = self.kernel, theta = self.k_theta, l = self.k_l)
        
        if self.min_log:
            gp.best_fit(xtrain, ytrain)
        else:
            gp.fit(xtrain, ytrain)

        
        # Finding argmax of the acquisition function.
        x_max = self.acq_max(gp, ymax, restarts, self.bounds)
                

        for i in range(num_it):
            op_start = datetime.now()

            xtrain = numpy.concatenate((xtrain, x_max.reshape((1, self.dim))), axis = 0)
            ytrain = numpy.concatenate((ytrain, self.f(x_max).reshape(1)), axis = 0)

            ymax = ytrain.max()

            #Updating the GP.
            if self.min_log:
                gp.best_fit(xtrain, ytrain)
            else:
                gp.fit(xtrain, ytrain)

            # Finding new argmax of the acquisition function.
            x_max = self.acq_max(gp, ymax, restarts, self.bounds)

            pi.print_info(op_start, i, x_max, ymax, xtrain, ytrain)
            
                
        if full_out:
            return ytrain.max(), xtrain[numpy.argmax(ytrain)], ytrain, xtrain
        else:
            return ytrain.max(), xtrain[numpy.argmax(ytrain)]


    def log_maximize(self, init_points = 3, restarts = 10, num_it = 15, verbose = 2, full_out = False):
        '''This method performs the optimization algorithm in a log scale of the bounds provided. Particularly useful when
           the order of magnitude of the bounds is too different.'''


        # ------------------------------ // ------------------------------ // ------------------------------ #
        for n, pair in enumerate(self.bounds):
            if pair[0] <= 0:
                raise RuntimeError('The lower bound of parameter %i is less or equal to zero, log grid requires strictly positive lower bounds.' % n)

        #Put all the bounds in the 0-1 interval of a log scale.
        self.log_bounds = numpy.log10(self.bounds/self.bounds[:, [0]]) / numpy.log10(self.bounds/self.bounds[:, [0]])[:, [1]]
        min_max_ratio = numpy.log10(self.bounds[:, 1] / self.bounds[:, 0])
        xmins = self.bounds[:, 0]


        if not self.user_init:
            print('Optimization procedure is initializing at %i random points.' % init_points)
            xtrain = numpy.asarray([numpy.random.uniform(x[0], x[1], size = init_points) for x in self.log_bounds]).T
            ytrain = numpy.asarray([self.f(xmins * (10 ** (x * min_max_ratio))) for x in xtrain])
            print('Optimization procedure is done initializing.')
        else:
            print('Optimization procedure is initializing at %i random points.' % init_points)
            new_xtrain = numpy.asarray([numpy.random.uniform(x[0], x[1], size = init_points) for x in self.log_bounds]).T
            
            xtrain = numpy.concatenate((numpy.log10(self.user_x/self.bounds[:, [0]]) / numpy.log10(self.bounds/self.bounds[:, [0]])[:, [1]],\
                                        new_xtrain), axis = 0)
            ytrain = numpy.concatenate((self.user_y, numpy.asarray([self.f(xmins * (10 ** (x * min_max_ratio))) for x in new_xtrain])))
            print('Optimization procedure is done initializing.')


        pi = print_info(verbose)
        ymax = ytrain.max()


        # ------------------------------ // ------------------------------ // ------------------------------ #
        # Fitting the gaussian process
        gp = GP(kernel = self.kernel, theta = self.k_theta, l = self.k_l)
        if self.min_log:
            gp.best_fit(xtrain, ytrain)
        else:
            gp.fit(xtrain, ytrain)
        

        # Finding argmax of the acquisition function.
        x_max = self.acq_max(gp, ymax, restarts, self.log_bounds)

        for i in range(num_it):
            op_start = datetime.now()

            xtrain = numpy.concatenate((xtrain, x_max.reshape((1, self.dim))), axis = 0)
            ytrain = numpy.concatenate((ytrain, self.f(xmins * (10 ** (x_max * min_max_ratio))).reshape(1)), axis = 0)

            ymax = ytrain.max()

            #Updating the GP.
            if self.min_log:
                gp.best_fit(xtrain, ytrain)
            else:
                gp.fit(xtrain, ytrain)


            # Finding new argmax of the acquisition function.
            x_max = self.acq_max(gp, ymax, restarts, self.log_bounds)

            pi.print_log(op_start, i, x_max, ymax, xtrain, ytrain)
                

        if full_out:
            return ytrain.max(), xmins * (10 ** (xtrain[numpy.argmax(ytrain)] * min_max_ratio)), ytrain, xtrain
        else:
            return ytrain.max(), xmins * (10 ** (xtrain[numpy.argmax(ytrain)] * min_max_ratio))



    # ------------------------------ // ------------------------------ # ------------------------------ // ------------------------------ #
    # ------------------------------ // ------------------------------ # ------------------------------ // ------------------------------ #
    def initialize(self, points):
        '''The user can pass a colection of points to initialize the object and
        point the algorithm in the right direction.'''
        

        print('Initializing %i points...' % len(points))
        op_start = datetime.now()

        user_pts = numpy.asarray(points).reshape((len(points), self.dim))

        if self.user_init == False:
    
            self.user_x = user_pts
            self.user_y = numpy.asarray([self.f(x) for x in user_pts])

        else:
            self.user_x = numpy.concatenate((self.user_x, numpy.asarray(points).reshape((len(points), self.dim))), axis = 0)
            self.user_y = numpy.concatenate((self.user_y, numpy.asarray([self.f(x) for x in user_pts])))

        numpy.set_printoptions(precision = 4, suppress = True)
        minutes, seconds = divmod((datetime.now() - op_start).seconds, 60)
        print('...done in %s:%s' % (minutes,seconds))
        print('The current maximum is: %f, at position: ' % numpy.max(self.user_y), self.user_x[numpy.argmax(self.user_y)])


        self.user_init = True
