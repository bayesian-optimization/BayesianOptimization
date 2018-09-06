"""Example of how to use this bayesian optimization package."""

import sys
sys.path.append("./")
from bayes_opt import BayesianOptimization

# Lets find the maximum of a simple quadratic function of two variables
# We create the bayes_opt object and pass the function to be maximized
# together with the parameters names and their bounds.
def f(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1
bo = BayesianOptimization({'x': (-4, 4), 'y': (-3, 3)})

# One of the things we can do with this object is pass points
# which we want the algorithm to probe. A dictionary with the
# parameters names and a list of values to include in the search
# must be given.
bo.explore({'x': [-1, 3], 'y': [-2, 2]})

# Additionally, if we have any prior knowledge of the behaviour of
# the target function (even if not totally accurate) we can also
# tell that to the optimizer.
# Here we pass a dictionary with 'target' and parameter names as keys and a
# list of corresponding values
bo.initialize(
    {
        'target': [-1, -1],
        'x': [1, 1],
        'y': [0, 2]
    }
)

# Once we are satisfied with the initialization conditions
# we let the algorithm do its magic by calling the maximize()
# method.

xs = bo.acquire_init(init_points=5)
bo.init(xs, [f(*x) for x in xs])
n_iter=15
for i in range(n_iter):
    x = bo.acquire(kappa=2)
    y = f(*x)
    bo.observe(x, y)

# The output values can be accessed with self.res
print(bo.res['max'])

## If we are not satisfied with the current results we can pickup from
## where we left, maybe pass some more exploration points to the algorithm
## change any parameters we may choose, and the let it run again.
#bo.explore({'x': [0.6], 'y': [-0.23]})

## Making changes to the gaussian process can impact the algorithm
## dramatically.
#gp_params = {'kernel': None,
#             'alpha': 1e-5}
#
## Run it again with different acquisition function
#bo.maximize(n_iter=5, acq='ei', **gp_params)
#
## Finally, we take a look at the final results.
#print(bo.res['max'])
#print(bo.res['all'])
