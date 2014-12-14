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

__author__ = 'fnogueira'


import numpy
from datetime import datetime

from sklearn.gaussian_process import GaussianProcess as GP

from scipy.optimize import minimize
from .helpers import acquisition, print_info



# ----------------------- // ----------------------- # ----------------------- // ----------------------- #
def acq_max(ac, gp, ymax, restarts, bounds):
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

    x_max = bounds[:, 0]
    ei_max = 0

    for i in range(restarts):
        #Sample some points at random.
        x_try = numpy.asarray([numpy.random.uniform(x[0], x[1], size=1) for x in bounds]).T

        #Find the minimum of minus que acquisition function
        res = minimize(lambda x: -ac(x, gp=gp, ymax=ymax), x_try, bounds=bounds, method='L-BFGS-B')

        #Store it if better than previous minimum(maximum).
        if -res.fun >= ei_max:
            x_max = res.x
            ei_max = -res.fun

    return x_max

def unique_rows(a):
    '''
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    '''

    # Sort array and kep track of where things should go back to
    order = numpy.lexsort(a.T)
    reorder = numpy.argsort(order)

    a = a[order]
    diff = numpy.diff(a, axis=0)
    ui = numpy.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


###

class bayes_opt(object):

    def __init__(self, f, pbounds, verbose=1):


        # Store the original dictionary
        self.pbounds = pbounds

        # Get the name of the parameters
        self.keys = list(pbounds.keys())

        # Find number of parameters
        self.dim = len(pbounds)

        # Create an array with parameters bounds
        self.bounds = []
        for key in pbounds.keys():
            self.bounds.append(pbounds[key])
        self.bounds = numpy.asarray(self.bounds)

        # Some function to be optimized
        self.f = f

        # Initialization lists
        self.initialized = False

        self.init_points = []
        self.x_init = []
        self.y_init = []

        # Verbose
        self.verbose = verbose

    def init(self, init_points):

        # Generate random points
        l = [numpy.random.uniform(x[0], x[1], size=init_points) for x in self.bounds]

        # Concatenate its transpose to the list of init points
        self.init_points += list(map(list, zip(*l)))

        # Create empty list to store the new values of the function
        y_init = []

        for x in self.init_points:

            if self.verbose:
                print('Initializing function at point: ', dict(zip(self.keys, x)), end='')

            y_init.append(self.f(**dict(zip(self.keys, x))))

            if self.verbose:
                print(' | result: %f' % y_init[-1])

        self.init_points += self.x_init

        y_init += self.y_init
        y_init = numpy.asarray(y_init)

        self.X = numpy.asarray(self.init_points)
        self.Y = y_init

        self.initialized = True

    # ------------------------------ // ---- # ----- // ------------------------------ #
    # ------------------------------ // ---- # ----- // ------------------------------ #
    def explore(self, points_dict):
        ''' Main optimization method.
            Parameters
            ----------
            points_dict: {p1: [x1, x2...], p2: [y1, y2, ...]}

            Returns
            -------
            Nothing.

        '''

        ################################################
        # Consistency check
        param_tup_lens = []

        for key in self.keys:
            param_tup_lens.append(len(list(points_dict[key])))

        if all([e == param_tup_lens[0] for e in param_tup_lens]):
            pass
        else:
            raise ValueError('The same number of initialization points must be entered for every parameter.')


        ################################################
        # Turn into list of lists

        all_points = []
        for key in self.keys:
            all_points.append(points_dict[key])

        # Take transpose of list
        self.init_points = list(map(list, zip(*all_points)))


    def initialize(self, points_dict):
        ''' Main optimization method.
            Parameters
            ----------
            points_dict: {y: {x1: x, ...}}


            Returns
            -------
            Nothing.

        '''
        ################################################
        # Turn into list of lists

        for target in points_dict:

            self.y_init.append(target)

            all_points = []
            for key in self.keys:
                all_points.append(points_dict[target][key])

            self.x_init.append(all_points)


    # ----------------------- // ----------------------- # ----------------------- // ----------------------- #
    def maximize(self, init_points=5, restarts=50, n_iter=25, acq='ei', **gp_params):
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
        # Start a timer
        total_time = datetime.now()

        # Create instance of printer object
        printI = print_info(self.verbose)

        # Set acquisition function
        AC = acquisition()
        ac_types = {'ei': AC.EI, 'pi': AC.PoI, 'ucb': AC.UCB}
        ac = ac_types[acq]

        # Initialize x, y and find current ymax
        if not self.initialized:
            self.init(init_points)

        ymax = self.Y.max()

        # ------------------------------ // ------------------------------ // ------------------------------ #
        # Fitting the gaussian process.
        gp = GP(theta0=0.01*numpy.ones(self.dim),\
                thetaU=0.2*numpy.ones(self.dim),\
                thetaL=0.0005*numpy.ones(self.dim),\
                random_start=15)

        gp.set_params(**gp_params)

        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        gp.fit(self.X[ur], self.Y[ur])


        # Finding argmax of the acquisition function.
        x_max = acq_max(ac, gp, ymax, restarts, self.bounds)


        for i in range(n_iter):
            op_start = datetime.now()

            # Append most recently generated values to X and Y arrays
            self.X = numpy.concatenate((self.X, x_max.reshape((1, self.dim))), axis=0)
            self.Y = numpy.append(self.Y, self.f(**dict(zip(self.keys, x_max))))

            #Updating the GP.
            ur = unique_rows(self.X)
            gp.fit(self.X[ur], self.Y[ur])

            # Finding new maximum value to search for next probe point.
            ymax = self.Y.max()

            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac, gp, ymax, restarts, self.bounds)

            # Print stuff
            printI.print_info(op_start, i, x_max, ymax, self.X, self.Y, self.keys)


        self.res = {}
        self.res['max'] = {'max_val': self.Y.max(), 'max_params': dict(zip(self.keys, self.X[self.Y.argmax()]))}
        self.res['all'] = {'values': [], 'params': []}


        for t, p in zip(self.Y, self.X):
            self.res['all']['values'].append(t)
            self.res['all']['params'].append(dict(zip(self.keys, p)))


        if self.verbose:
            tmin, tsec = divmod((datetime.now() - total_time).total_seconds(), 60)
            print('Optimization finished with maximum: %8f, at position: %8s.' % (self.res['max']['max_val'],\
                                                                                  self.res['max']['max_params']))
            print('Time taken: %i minutes and %s seconds.' % (tmin, tsec))


