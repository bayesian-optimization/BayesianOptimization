'''
________ Bayesian optimization ________

Issues:

See papers: http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
            http://arxiv.org/pdf/1012.2599v1.pdf
            http://www.gaussianprocess.org/gpml/

for references.

Fernando Nogueira
'''

from __future__ import print_function
from __future__ import division



import numpy
from datetime import datetime

from scipy.optimize import minimize

from math import exp, fabs, sqrt, log, pi

from GP import GP
from support.objects import covariance, sample_covariance, kernels, acquisition, print_info



################################################################################
##############################____Bayes_Class____###############################
################################################################################

class bayes_opt:
    '''
    ________ Bayesian Optimization Class ________

    An object to perform global constrained optimization.
    

    Parameters
    ----------

    f : The function whose maximum is to be found. It must be of the form f(params) where params
        is an 1d-array.
        --- Given a function F(a, b, c, ...) of N variables, a dictionary with the bounds for each variable
        such as {'a' : (0, 1), 'b' : (10, 542), ...} should be passed to the object. ---


    params_dict : The minimum and maximum bounds for the variables of the target function. It has to be a
                  dictionary with keys corresponding to the functions arguments for each bound tupple.
                  e.g.: {'a' : (0, 1), 'b' : (10, 542), ...}

    kernel : defaults to 'squared_exp', is the kernel to be used in the gaussian process.

    acq : defaults to 'ei' (Expected Improvement), is the acquisition function to be used
          when deciding where to sample next.

    min_log : Parameter dictating whether to find the kernel parameters that lead to the best gp fit
              (maximum likelihood) or to use the specified kernel parameters.


    Member Functions
    ----------------

    set_acquisition : Member function to set the acquisition function to be used. Currently implemented
                      options are PoI, Probability of Improvement; EI, Expected Improvement; and UCB, upper
                      confidence bound, it takes the parameter of the UCB, k, as argument (defaults to 1).

    set_kernel : Member function to set the kernel function to be used. Similar as the for the GP class.

    acq_max : A member function to find the maximum of the acquisition function. It takes a GP object and
              the number os restarts as additional arguments. It uses the scipy object minimize with method 'L-BFGS-B'
              to find the local minima of minus the acquisition function. It reapeats it a number of times to avoid
              falling into local minima.

    maximize : One of the two main methods of this object. It performs bayesian optimization and return the
               maximum value of the function together with the position of the maximum. A full_output option can be
               turned on to have the object return all the sampled values of X and Y.

    log_maximize : The other main method of this object, behaves similarly to maximize, however it performs
                   optimization on a log scale of the arguments. This is particularly useful for when the order of magnitude
                   of the maximum bound is much greater than that of the minimum bound. Should be the prefered method for when
                   optimizing the parameters of say, a classifier in the range (0.001, 100), for example.

    initialize : This member function add to the collection of sampled points used by both maximize methods user
                 defined points. It allow the user to have some control over the sampling space, as well as guide the
                 optimizer in the right direction for cases when a number of relevant points are known.
                 A dictionary with values (single or multiple) for each argument should be provided.
                 {'a' : (0, 1, 0.5, 0.4,...), 'b' : (10, 542, 222, 128,...), ...}

    '''

    def __init__(self, f, params_dict, kernel = 'squared_exp', acq = 'ei', min_log = True):
        '''This is an object to find the global maximum of an unknown function via gaussian processes./n
           It takes a function of N variables and the lower and upper bounds for each variable as parameters.
           The function passed to this object should take as array as entry, therefore a function F of N
           variables should be passed as, f = lambda x: F(x[0],...,x[N-1]).

           Member variables
           ----------------

           ##

           self.kernel : Stores the kernel of choice as a member variable.

           self.k_theta : Stores the parameter theta of the kernel.

           self.k_l : Stores the parameter l of the kernel.

           self.ac : Stores the acquisition function of choice as a member variable.

           ##

           self.keys : Holds the keys of params_dict, which has the variables names in it.

           self.pdict : Stores params_dict

           self.bounds : Stores the variables bounds as a numpy array.

           self.log_bounds : A member variable to store the log scaled bounds, only used if log_minimize is used
                             and the minimum bound is greater than zero for all variables.

           self.dim : A member variable that stores the dimension of the target function.

           ##

           self.user_x : Member variable used to store x values passed to the initialize method.

           self.user_y : Member variable used to store f(x) values passed to the initialize method.

           self.user_init : A member variable that keeps track of whether the method 'initialize' has been called.

           self.min_log : Member variable to store whether maximum likelihood is to be used when fitting the
                          gp or not.


        '''

        self.keys = list(params_dict.keys())
        self.pdict = params_dict
        self.dim = len(params_dict)

        self.bounds = []
        for key in params_dict.keys():
            self.bounds.append(params_dict[key])

        self.bounds = numpy.asarray(self.bounds)

        self.log_bounds = 0 * numpy.asarray(self.bounds)

        # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
        for n, pair in enumerate(self.bounds):
            if pair[1] == pair[0]:
                raise RuntimeError('The upper and lower bound of parameter %i are the same, the upper bound must be greater than the lower bound.' % n)
            if pair[1] < pair[0]:
                raise RuntimeError('The upper bound of parameter %i is less than the lower bound, the upper bound must be greater than the lower bound.' % n)

        
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
        self.user_x = numpy.empty((1, len(params_dict)))
        self.user_y = numpy.empty(1)


        
    # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
    # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
    def set_acquisition(self, acq = 'ucb', k = 1):
        ''' Set a new acquisition function.

            Parameters
            ----------
            acq : One of the supported acquisition function names or a custom one.

            k : Parameter k of the UCB acquisition function.
  
            Returns
            -------
            Nothing.
        '''
        
        ac = acquisition(k)
        ac_types = {'ei' : ac.EI, 'pi' : ac.PoI, 'ucb' : ac.UCB}
        try:
            self.ac = ac_types[acq]
        except KeyError:
            print('Custom acquisition function being used.')
            self.ac = acq

    def set_kernel(self, kernel = 'ARD_matern', theta = 1, l = 1):
        ''' Set a new kernel for the gaussian process.

            Parameters
            ----------
            kernel : One of the supported kernel names or a custom kernel function.

            theta : Theta parameter of the kernel.

            l : l parameter of the kernel.
  
            Returns
            -------
            Nothing.
        '''
        
        self.kernel = kernel
        self.k_theta = theta
        self.k_l = l

    # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
    def acq_max(self, gp, ymax, restarts, Bounds):
        ''' A function to find the maximum of the acquisition function using the 'L-BFGS-B' method.

            Parameters
            ----------
            gp : A gaussian process fitted to the relevant data.

            ymax : The current maximum known value of the target function.

            restarts : The number of times minimation if to be repeated. Larger number of restarts
                       improves the chances of finding the true maxima.

            Bounds : The variables bounds to limit the search of the acq max.
            
  
            Returns
            -------
            x_max : The arg max of the acquisition function.
        '''

        x_max = Bounds[:, 0]
        ei_max = 0

        for i in range(restarts):
            #Sample some points at random.
            x_try = numpy.asarray([numpy.random.uniform(x[0], x[1], size = 1) for x in Bounds]).T
            
            #Find the minimum of minus que acquisition function
            res = minimize(lambda x: -self.ac(x, gp = gp, ymax = ymax), x_try, bounds = Bounds, method = 'L-BFGS-B')

            #Store it if better than previous minimum(maximum).
            if -res.fun >= ei_max:
                x_max = res.x
                ei_max = -res.fun

        return x_max


    # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
    # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
    def maximize(self, init_points = 3, restarts = 10, num_it = 15, verbose = 2, full_out = False):
        ''' Main optimization method.

            Parameters
            ----------
            init_points : Number of randomly chosen points to sample the target function before fitting the gp.

            restarts : The number of times minimation if to be repeated. Larger number of restarts
                       improves the chances of finding the true maxima.

            num_it : Total number of times the process is to reapeated. Note that currently this methods does not have
                     stopping criteria (due to a number of reasons), therefore the total number of points to be sampled
                     must be specified.

            verbose : The amount of information to be printed during optimization. Accepts 0(nothing), 1(partial), 2(full).

            full_out : If the full output is to be returned or just the function maximum and arg max.
            
  
            Returns
            -------
            y_max, x_max : The function maximum and its position.

            y_max, x_max, y, x : In addition to the maximum and arg max, return all the sampled x and y points.
            
        '''
        total_time = datetime.now()

        # ------------------------------ // ------------------------------ // ------------------------------ #
        if not self.user_init:
            print('Optimization procedure is initializing at %i random points.' % init_points)
            #Chose some points at random
            xtrain = numpy.asarray([numpy.random.uniform(x[0], x[1], size = init_points) for x in self.bounds]).T
            #Pass a dictionary with random parameters values back to the function.
            ytrain = numpy.asarray([self.f(**dict(zip(self.keys, xs))) for xs in xtrain])            
            print('Optimization procedure is done initializing.')
        else:
            print('Optimization procedure is initializing at %i random points.' % init_points)
            new_xtrain = numpy.asarray([numpy.random.uniform(x[0], x[1], size = init_points) for x in self.bounds]).T
            
            xtrain = numpy.concatenate((self.user_x, new_xtrain), axis = 0)
            ytrain = numpy.concatenate((self.user_y, numpy.asarray([self.f(**dict(zip(self.keys, xs))) for xs in new_xtrain])))
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
            #ytrain = numpy.concatenate((ytrain, self.f(**dict(zip(self.keys, x_max)))), axis = 0)
            ytrain = numpy.append(ytrain, self.f(**dict(zip(self.keys, x_max))))

            ymax = ytrain.max()

            #Updating the GP.
            if self.min_log:
                gp.best_fit(xtrain, ytrain)
            else:
                gp.fit(xtrain, ytrain)

            # Finding new argmax of the acquisition function.
            x_max = self.acq_max(gp, ymax, restarts, self.bounds)
            # Printing everything
            pi.print_info(op_start, i, x_max, ymax, xtrain, ytrain, self.keys)
            

        tmin, tsec = divmod((datetime.now() - total_time).total_seconds(), 60)
        print('Optimization finished with maximum: %8f | Time taken: %i minutes and %s seconds' % (ytrain.max(), tmin, tsec))
                
        if full_out:
            return ytrain.max(), dict(zip(self.keys, xtrain[numpy.argmax(ytrain)])), ytrain, xtrain
        else:
            return ytrain.max(), dict(zip(self.keys, xtrain[numpy.argmax(ytrain)]))
        

    def log_maximize(self, init_points = 3, restarts = 10, num_it = 15, verbose = 2, full_out = False):
        ''' Main optimization method perfomed in a log scale.

            Parameters
            ----------
            init_points : Number of randomly chosen points to sample the target function before fitting the gp.

            restarts : The number of times minimation if to be repeated. Larger number of restarts
                       improves the chances of finding the true maxima.

            num_it : Total number of times the process is to reapeated. Note that currently this methods does not have
                     stopping criteria (due to a number of reasons), therefore the total number of points to be sampled
                     must be specified.

            verbose : The amount of information to be printed during optimization. Accepts 0(nothing), 1(partial), 2(full).

            full_out : If the full output is to be returned or just the function maximum and arg max.
            
  
            Returns
            -------
            y_max, x_max : The function maximum and its position.

            y_max, x_max, y, x : In addition to the maximum and arg max, return all the sampled x and y points.
            
        '''
        total_time = datetime.now()


        # ------------------------------ // ------------------------------ // ------------------------------ #
        for n, pair in enumerate(self.bounds):
            if pair[0] <= 0:
                raise RuntimeError('The lower bound of parameter %i is less or equal to zero, log grid requires strictly positive lower bounds.' % n)

        #Put all the bounds in the 0-1 interval of a log scale.
        self.log_bounds = numpy.log10(self.bounds/self.bounds[:, [0]]) / numpy.log10(self.bounds/self.bounds[:, [0]])[:, [1]]
        min_max_ratio = numpy.log10(self.bounds[:, 1] / self.bounds[:, 0])
        xmins = self.bounds[:, 0]

        def return_log(x):
            return xmins * (10 ** (x * min_max_ratio))


        if not self.user_init:
            print('Optimization procedure is initializing at %i random points.' % init_points)
            #Sampling some points are random to define xtrain.
            xtrain = numpy.asarray([numpy.random.uniform(x[0], x[1], size = init_points) for x in self.log_bounds]).T
            ytrain = numpy.asarray([self.f(**dict(zip(self.keys, return_log(x)))) for x in xtrain])
            print('Optimization procedure is done initializing.')
        else:
            print('Optimization procedure is initializing at %i random points.' % init_points)
            #Sampling some points are random
            new_xtrain = numpy.asarray([numpy.random.uniform(x[0], x[1], size = init_points) for x in self.log_bounds]).T
            #Setting the user init points in the log scale
            init_x = numpy.log10(self.user_x/self.bounds[:, 0]) / numpy.log10(self.bounds[:, 1]/self.bounds[:, 0])
            #Defining xtrain and ytrain
            xtrain = numpy.concatenate((init_x, new_xtrain), axis = 0)
            ytrain = numpy.concatenate((self.user_y, numpy.asarray([self.f(**dict(zip(self.keys, return_log(x)))) for x in new_xtrain])))

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
            #ytrain = numpy.concatenate((ytrain, self.f(**dict(zip(self.keys, return_log(x_max))))), axis = 0)
            ytrain = numpy.append(ytrain, self.f(**dict(zip(self.keys, return_log(x_max)))))
            
            ymax = ytrain.max()

            #Updating the GP.
            if self.min_log:
                gp.best_fit(xtrain, ytrain)
            else:
                gp.fit(xtrain, ytrain)


            # Finding new argmax of the acquisition function.
            x_max = self.acq_max(gp, ymax, restarts, self.log_bounds)
            # Printing everything
            pi.print_log(op_start, i, x_max, xmins, min_max_ratio, ymax, xtrain, ytrain, self.keys)
                

        tmin, tsec = divmod((datetime.now() - total_time).total_seconds(), 60)
        print('Optimization finished with maximum: %8f | Time taken: %i minutes and %s seconds' % (ytrain.max(), tmin, tsec))

        if full_out:
            return ytrain.max(), dict(zip(self.keys, return_log(xtrain[numpy.argmax(ytrain)]))), ytrain, return_log(xtrain)
        else:
            return ytrain.max(), dict(zip(self.keys, return_log(xtrain[numpy.argmax(ytrain)])))




    # ------------------------------ // ------------------------------ # ------------------------------ // ------------------------------ #
    # ------------------------------ // ------------------------------ # ------------------------------ // ------------------------------ #
    def initialize(self, points_dict):
        ''' Main optimization method.

            Parameters
            ----------
            points : The collection of points to use as part of the initialization.
            
  
            Returns
            -------
            Nothing.
            
        '''

        
        list_points = []
        for key in self.pdict.keys():
            try:
                list_points.append(points_dict[key])
            except KeyError:
                print('Parameters names have to match exactly.')
        

        list_points = numpy.asarray(list_points)

        try:
            list_points.shape[1]
            user_pts = list_points.T
        except:
            user_pts = list_points.reshape((1, self.dim))


        print('Initializing %i points...' % len(user_pts))
        op_start = datetime.now()

        if self.user_init == False:
    
            self.user_x = user_pts
            self.user_y = numpy.asarray([self.f(**dict(zip(self.keys, x))) for x in user_pts])

        else:
            self.user_x = numpy.concatenate((self.user_x, numpy.asarray(list_points).reshape((len(user_pts), self.dim))), axis = 0)
            self.user_y = numpy.concatenate((self.user_y, numpy.asarray([self.f(**dict(zip(self.keys, x))) for x in user_pts])))

        numpy.set_printoptions(precision = 4, suppress = True)
        minutes, seconds = divmod((datetime.now() - op_start).total_seconds(), 60)
        
        print('...done in %s minutes and %s seconds.' % (minutes, seconds))
        print('The current maximum is: %f, with parameters: ' % numpy.max(self.user_y), dict(zip(self.keys, self.user_x[numpy.argmax(self.user_y)])))


        self.user_init = True



    def init(self, points_dict):
        '''A function to perform all initialization and clear the optimize methods - To be constructed'''
        return 0
