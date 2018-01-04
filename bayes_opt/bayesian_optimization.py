from __future__ import print_function
from __future__ import division

import numpy as np
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from .helpers import (UtilityFunction, PrintLog, acq_max, ensure_rng)
from .target_space import TargetSpace


class BayesianOptimization(object):

    def __init__(self, f, pbounds, random_state=None, verbose=1):
        """
        :param f:
            Function to be maximized.

        :param pbounds:
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        :param verbose:
            Whether or not to print progress.

        """
        # Store the original dictionary
        self.pbounds = pbounds

        self.random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self.space = TargetSpace(f, pbounds, random_state)

        # Initialization flag
        self.initialized = False

        # Initialization lists --- stores starting points before process begins
        self.init_points = []
        self.x_init = []
        self.y_init = []

        # Counter of iterations
        self.i = 0

        # Internal GP regressor
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=25,
            random_state=self.random_state
        )

        # Utility Function placeholder
        self.util = None

        # PrintLog object
        self.plog = PrintLog(self.space.keys)

        # Output dictionary
        self.res = {}
        # Output dictionary
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values': [], 'params': []}

        # non-public config for maximizing the aquisition function
        # (used to speedup tests, but generally leave these as is)
        self._acqkw = {'n_warmup': 100000, 'n_iter': 250}

        # Verbose
        self.verbose = verbose

    def init(self, init_points):
        """
        Initialization method to kick start the optimization process. It is a
        combination of points passed by the user, and randomly sampled ones.

        :param init_points:
            Number of random points to probe.
        """
        # Concatenate new random points to possible existing
        # points from self.explore method.
        rand_points = self.space.random_points(init_points)
        self.init_points.extend(rand_points)

        # Evaluate target function at all initialization points
        for x in self.init_points:
            y = self._observe_point(x)

        # Add the points from `self.initialize` to the observations
        if self.x_init:
            x_init = np.vstack(self.x_init)
            y_init = np.hstack(self.y_init)
            for x, y in zip(x_init, y_init):
                self.space.add_observation(x, y)
                if self.verbose:
                    self.plog.print_step(x, y)

        # Updates the flag
        self.initialized = True

    def _observe_point(self, x):
        y = self.space.observe_point(x)
        if self.verbose:
            self.plog.print_step(x, y)
        return y

    def explore(self, points_dict, eager=False):
        """Method to explore user defined points.

        :param points_dict:
        :param eager: if True, these points are evaulated immediately
        """
        if eager:
            self.plog.reset_timer()
            if self.verbose:
                self.plog.print_header(initialization=True)

            points = self.space._dict_to_points(points_dict)
            for x in points:
                self._observe_point(x)
        else:
            points = self.space._dict_to_points(points_dict)
            self.init_points = points

    def initialize(self, points_dict):
        """
        Method to introduce points for which the target function value is known

        :param points_dict:
            dictionary with self.keys and 'target' as keys, and list of
            corresponding values as values.

        ex:
            {
                'target': [-1166.19102, -1142.71370, -1138.68293],
                'alpha': [7.0034, 6.6186, 6.0798],
                'colsample_bytree': [0.6849, 0.7314, 0.9540],
                'gamma': [8.3673, 3.5455, 2.3281],
            }

        :return:
        """

        self.y_init.extend(points_dict['target'])
        for i in range(len(points_dict['target'])):
            all_points = []
            for key in self.space.keys:
                all_points.append(points_dict[key][i])
            self.x_init.append(all_points)

    def initialize_df(self, points_df):
        """
        Method to introduce point for which the target function
        value is known from pandas dataframe file

        :param points_df:
            pandas dataframe with columns (target, {list of columns matching
            self.keys})

        ex:
              target        alpha      colsample_bytree        gamma
        -1166.19102       7.0034                0.6849       8.3673
        -1142.71370       6.6186                0.7314       3.5455
        -1138.68293       6.0798                0.9540       2.3281
        -1146.65974       2.4566                0.9290       0.3456
        -1160.32854       1.9821                0.5298       8.7863

        :return:
        """

        for i in points_df.index:
            self.y_init.append(points_df.loc[i, 'target'])

            all_points = []
            for key in self.space.keys:
                all_points.append(points_df.loc[i, key])

            self.x_init.append(all_points)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        :param new_bounds:
            A dictionary with the parameter name and its new bounds

        """
        # Update the internal object stored dict
        self.pbounds.update(new_bounds)
        self.space.set_bounds(new_bounds)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 xi=0.0,
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
            Acquisition function to be used, defaults to Upper Confidence Bound.

        :param gp_params:
            Parameters to be passed to the Scikit-learn Gaussian Process object

        Returns
        -------
        :return: Nothing

        Example:
        >>> xs = np.linspace(-2, 10, 10000)
        >>> f = np.exp(-(xs - 2)**2) + np.exp(-(xs - 6)**2/10) + 1/ (xs**2 + 1)
        >>> bo = BayesianOptimization(f=lambda x: f[int(x)],
        >>>                           pbounds={"x": (0, len(f)-1)})
        >>> bo.maximize(init_points=2, n_iter=25, acq="ucb", kappa=1)
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

        y_max = self.space.Y.max()

        # Set parameters if any was passed
        self.gp.set_params(**gp_params)

        # Find unique rows of X to avoid GP from breaking
        self.gp.fit(self.space.X, self.space.Y)

        # Finding argmax of the acquisition function.
        x_max = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        y_max=y_max,
                        bounds=self.space.bounds,
                        random_state=self.random_state,
                        **self._acqkw)

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
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = False
            while x_max in self.space:
                x_max = self.space.random_points(1)[0]
                pwarning = True

            # Append most recently generated values to X and Y arrays
            y = self.space.observe_point(x_max)
            if self.verbose:
                self.plog.print_step(x_max, y, pwarning)

            # Updating the GP.
            self.gp.fit(self.space.X, self.space.Y)

            # Update the best params seen so far
            self.res['max'] = self.space.max_point()
            self.res['all']['values'].append(y)
            self.res['all']['params'].append(dict(zip(self.space.keys, x_max)))

            # Update maximum value to search for next probe point.
            if self.space.Y[-1] > y_max:
                y_max = self.space.Y[-1]

            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp,
                            y_max=y_max,
                            bounds=self.space.bounds,
                            random_state=self.random_state,
                            **self._acqkw)

            # Keep track of total number of iterations
            self.i += 1

        # Print a final report if verbose active.
        if self.verbose:
            self.plog.print_summary()

    def points_to_csv(self, file_name):
        """
        After training all points for which we know target variable
        (both from initialization and optimization) are saved

        :param file_name: name of the file where points will be saved in the csv
            format

        :return: None
        """

        points = np.hstack((self.space.X, np.expand_dims(self.space.Y, axis=1)))
        header = ','.join(self.space.keys + ['target'])
        np.savetxt(file_name, points, header=header, delimiter=',', comments='')

    # --- API compatibility ---

    @property
    def X(self):
        warnings.warn("use self.space.X instead", DeprecationWarning)
        return self.space.X

    @property
    def Y(self):
        warnings.warn("use self.space.Y instead", DeprecationWarning)
        return self.space.Y

    @property
    def keys(self):
        warnings.warn("use self.space.keys instead", DeprecationWarning)
        return self.space.keys

    @property
    def f(self):
        warnings.warn("use self.space.target_func instead", DeprecationWarning)
        return self.space.target_func

    @property
    def bounds(self):
        warnings.warn("use self.space.dim instead", DeprecationWarning)
        return self.space.bounds

    @property
    def dim(self):
        warnings.warn("use self.space.dim instead", DeprecationWarning)
        return self.space.dim
