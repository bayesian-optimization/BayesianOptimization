import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction


def f(x):
    return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1/ (x ** 2 + 1)


if __name__ == '__main__':
    optimizer = BayesianOptimization(f=None, pbounds={'x': (-2, 2)}, verbose=2, random_state=1,
                                     allow_duplicate_points=True)
    optimizer.set_gp_params(normalize_y=True, alpha=2.5e-3, n_restarts_optimizer=20)  # tuning of the gaussian parameters...
    utility = UtilityFunction(kind="ucb", kappa=5, xi=1)  # kappa determines explore/Exploitation ratio
    for point in range(20):
        next_point_to_probe = optimizer.suggest(utility)
        NextPointValues = np.array(list(next_point_to_probe.values()))
        mean,std = optimizer._gp.predict(NextPointValues.reshape(1, -1),return_std=True)
        target = f(**next_point_to_probe)
        optimizer.register(params=next_point_to_probe, target=target)