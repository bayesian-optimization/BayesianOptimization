from bayes_opt import BayesianOptimization
'''
Example of how to use this bayesian optimization package.
'''

# Lets find the maximum of a simple quadratic function of two variables
# We create the bayes_opt object and pass the function to be maximized
# together with the paraters names and their bounds.
bo = BayesianOptimization(lambda x, y: -x**2 -(y-1)**2 + 1, {'x': (-4, 4), 'y': (-3, 3)})

# One of the things we can do with this object is pass points
# which we want the algorithm to probe. A dictionary with the
# parameters names and a list of values to include in the search
# must be given.
bo.explore({'x': [-1, 3], 'y': [-2, 2]})

# Additionally, if we have any prior knowledge of the behaviour of
# the target function (even if not totally accurate) we can also
# tell that to the optimizer.
# Here we pass a dictionary with target values as keys of another
# dictionary with parameters names and their corresponding value.
bo.initialize({-2: {'x': 1, 'y': 0}, -1.251: {'x': 1, 'y': 1.5}})

# Once we are satisfied with the initialization conditions
# we let the algorithm do its magic by calling the maximize()
# method.
bo.maximize(n_iter=15)

# The output values can be accessed with self.res
print(bo.res['max'])


# If we are not satisfied with the current results we can pickup from
# where we left, maybe pass some more exploration points to the algorithm
# change any parameters we may choose, and the let it run again.
bo.explore({'x': [0.6], 'y': [-0.23]})
bo.maximize(n_iter=5, acq='pi')

# Finally, we take a look at the final results.
print(bo.res['max'])
print(bo.res['all'])