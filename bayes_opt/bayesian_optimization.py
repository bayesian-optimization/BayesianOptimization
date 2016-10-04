from __future__ import print_function
from __future__ import division

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from .helpers import UtilityFunction, unique_rows, PrintLog, acq_max
__author__ = 'fmfn'


class BayesianOptimization(object):
    def __init__(self, f, pbounds, verbose=1, mapf=map):
        """
        :param f:
            Function to be maximized.

        :param pbounds:
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        :param verbose:
            Whether or not to print progress.

        :param mapf:
            Use a custom map function for function evaluation.
        """
        # Store the original dictionary
        self.pbounds = pbounds

        # Get the name of the parameters
        self.keys = list(pbounds.keys())

        # Find number of parameters
        self.dim = len(pbounds)

        # Create an array with parameters bounds
        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)

        # Some function to be optimized
        self.f = f

        # Initialization flag
        self.initialized = False

        # Initialization lists --- stores starting points before process begins
        self.init_points = []
        self.x_init = []
        self.y_init = []

        # Numpy array place holders
        self.X = None
        self.Y = None

        # Counter of iterations
        self.i = 0

        # Internal GP regressor
        self.gp = GaussianProcessRegressor(
            kernel=Matern(),
            n_restarts_optimizer=25,
        )

        # Utility Function placeholder
        self.util = None

        # PrintLog object
        self.plog = PrintLog(self.keys)

        # Output dictionary
        self.res = {}
        # Output dictionary
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values': [], 'params': []}

        # Initialise the map function to use for evaluation of f.
        self.mapf = mapf

        # Verbose
        self.verbose = verbose

    def _evalpoint(self, x):
        """
        Simply evaluates the objective function for x (private)

        :param x: model parameters

        :return: y
        """
        y = self.f(**dict(zip(self.keys, x)))

        if self.verbose:
            self.plog.print_step(x, y)

        return y

    def init(self, init_points):
        """
        Initialization method to kick start the optimization process. It is a
        combination of points passed by the user, and randomly sampled ones.

        :param init_points:
            Number of random points to probe.
        """

        # Generate random points
        l = [np.random.uniform(x[0], x[1], size=init_points) for x in self.bounds]

        # Concatenate new random points to possible existing
        # points from self.explore method.
        self.init_points += list(map(list, zip(*l)))

        # store init_points as self.X
        self.X = np.asarray(self.init_points)

        # there are no warnings for the initialisation, but we must keep track
        self.warnings = np.zeros(len(self.X))

        # Evaluate target function at all initialization points
        y_init = list(self.mapf(self._evalpoint, self.X))

        # Append any other points passed by the self.initialize method (these
        # also have a corresponding target value passed by the user).
        self.init_points += self.x_init

        # Append the target value of self.initialize method.
        y_init += self.y_init

        # save out target function values.
        self.Y = np.asarray(np.squeeze(y_init))

        # Updates the flag
        self.initialized = True

    def explore(self, points_dict):
        """
        Method to explore user defined points

        :param points_dict:
        :return:
        """

        # Consistency check
        param_tup_lens = []

        for key in self.keys:
            param_tup_lens.append(len(list(points_dict[key])))

        if all([e == param_tup_lens[0] for e in param_tup_lens]):
            pass
        else:
            raise ValueError('The same number of initialization points '
                             'must be entered for every parameter.')

        # Turn into list of lists
        all_points = []
        for key in self.keys:
            all_points.append(points_dict[key])

        # Take transpose of list
        self.init_points = list(map(list, zip(*all_points)))

    def initialize(self, points_dict):
        """
        Method to introduce point for which the target function
        value is known

        :param points_dict:
        :return:
        """

        for target in points_dict:

            self.y_init.append(target)

            all_points = []
            for key in self.keys:
                all_points.append(points_dict[target][key])

            self.x_init.append(all_points)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        :param new_bounds:
            A dictionary with the parameter name and its new bounds

        """

        # Update the internal object stored dict
        self.pbounds.update(new_bounds)

        # Loop through the all bounds and reset the min-max bound matrix
        for row, key in enumerate(self.pbounds.keys()):
            # Reset all entries, even if the same.
            self.bounds[row] = self.pbounds[key]

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 n_evals=1,
                 **gp_params):
        """
        Main optimization method.

        Parameters
        ----------
        :param init_points:
            Number of randomly chosen points to sample the
            target function before fitting the gp.

        :param n_iter:
            Total number of times the process is to repeated. Note that
            currently this methods does not have stopping criteria (due to a
            number of reasons), therefore the total number of points to be
            sampled must be specified.

        :param acq:
            Acquisition function to be used, defaults to Expected Improvement.

        :param n_evals:
            Maximum number of evaluations to run per tuning iteration (actual number depends on xi)

        :param gp_params:
            Parameters to be passed to the Scikit-learn Gaussian Process object


        Returns
        -------
        :return: Nothing
        """
        # Reset timer
        self.plog.reset_timer()

        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)

        # Initialize x, y and find current y_max
        if not self.initialized:
            if self.verbose:
                self.plog.print_header()
            self.init(init_points)

        y_max = self.Y.max()

        # Set parameters if any was passed
        self.gp.set_params(**gp_params)

        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])

        # Finding argmax of the acquisition function.

        x_max = np.asarray(
            [acq_max(ac=self.util.utility, gp=self.gp, y_max=y_max, bounds=self.bounds) for x in range(n_evals)])
        x_max = x_max[unique_rows(x_max)]  # prevent duplicates (this may mean less than n_evals bought forward)

        # Print new header
        if self.verbose:
            self.plog.print_header(initialization=False)
        # Iterative process of searching for the maximum. At each round the
        # most recent x and y values probed are added to the X and Y arrays
        # used to train the Gaussian Process. Next the maximum known value
        # of the target function is found and passed to the acq_max function.
        # The arg_max of the acquisition function is found and this will be
        # the next probed value of the target function in the next round.
        for i in range(n_iter):
            act_evals = len(x_max)  # record the actual number of evaluations in this round.

            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = np.zeros(act_evals)
            for k in range(act_evals):
                if np.any((self.X - x_max[k, :]).sum(axis=1) == 0):
                    x_max[k, :] = np.random.uniform(self.bounds[:, 0],
                                                    self.bounds[:, 1],
                                                    size=self.bounds.shape[0])

                    pwarning[k] = True

            # Append most recently generated values to X and Y arrays

            self.warnings = np.append(self.warnings, pwarning) # keep warnings for later..
            self.X = np.asarray(np.vstack((self.X, x_max)))
            new_y = list(self.mapf(self._evalpoint, x_max))
            self.Y = np.append(self.Y, np.asarray(new_y))
            # Updating the GP.
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])

            # Update maximum value to search for next probe point.
            if any(self.Y > y_max):
                y_max = self.Y.max()

            # Maximize acquisition function to find next probing point
            x_max = np.asarray([acq_max(ac=self.util.utility, gp=self.gp, y_max=y_max, bounds=self.bounds) for x in
                                range(n_evals)])
            x_max = x_max[unique_rows(x_max)]  # prevent duplicates (this may mean less than n_evals bought forward)

            # Spit the results into the .res strucure.
            self.res['max'] = {'max_val': self.Y.max(),
                               'max_params': dict(zip(self.keys,
                                                      self.X[self.Y.argmax()]))
                               }
            for k in range(len(self.X) - act_evals, len(self.X)):
                self.res['all']['values'].append(self.Y[k])
            self.res['all']['params'].append(dict(zip(self.keys, self.X[k])))

            # Print a final report if verbose active.
            if self.verbose:
                self.plog.print_summary()
