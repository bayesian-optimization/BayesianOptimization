from bayes_opt import BayesianOptimization

# function to be maximized - must find (x=0;y=10)
targetFunction = lambda x, y: -(x-0.5) ** 2 - (y - 10) ** 2 + 1

# define parameters bounds
bounds = {'y': (5, 15), 'x': (-3, 3)}
btypes = {'y':int, 'x':float}
bo = BayesianOptimization(targetFunction, bounds) #, ptypes=btypes)

bo.probe({"x": 1.4, "y": 6})
bo.probe({"x": 2.4, "y": 12})
bo.probe({"x": -2.4, "y": 13})

bo.maximize(init_points=5, n_iter=5, kappa=2)

# print results
print(f'Estimated position of the maximum: {bo.max}')
print(f'List of tested positions:\n{bo.res}')

