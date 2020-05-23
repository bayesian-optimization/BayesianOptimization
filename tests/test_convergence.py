import pytest
from bayes_opt import BayesianOptimization, UtilityFunction


def target_func(x):
    # arbitrary target func
    return x**2


PBOUNDS = {'x': (0, 20)}


def test_collision_in_convergence():
    optimizer = BayesianOptimization(None, PBOUNDS, random_state=1)
    util = UtilityFunction(kind='ucb', kappa=2.576, xi=0)
    with pytest.raises(KeyError):
        # Initial random search
        cache = []
        for i in range(5):
            params = optimizer.suggest(util)
            cache.append((params, target_func(**params)))
        for params, score in cache:
            optimizer.register(params, score)
        for i in range(100):
            params = optimizer.suggest(util)
            score = target_func(**params)
            optimizer.register(params, score)