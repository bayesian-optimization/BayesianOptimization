import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.logger import ScreenLogger


def f(x):
    return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1/ (x ** 2 + 1)

def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1



if __name__ == '__main__':
    repeat_points_version=True
    if repeat_points_version:
        optimizer = BayesianOptimization(f=f, pbounds={'x': (-2, 2)}, verbose=2, random_state=1, allow_duplicate_points=True)
        # logger = ScreenLogger()
        # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.set_gp_params(normalize_y=True, alpha=2.5e-3, n_restarts_optimizer=20)  # tuning of the gaussian parameters...
        utility = UtilityFunction(kind="ucb", kappa=5, xi=1)  # kappa determines explore/Exploitation ratio
        # optimizer.maximize(
        #     init_points=2,
        #     n_iter=20,
        # )

        optimizer.maximize(init_points=2,
                     n_iter=25,
                     utility_function=utility)
        # for point in range(20):
        #     next_point_to_probe = optimizer.suggest(utility)
        #     NextPointValues = np.array(list(next_point_to_probe.values()))
        #     mean,std = optimizer._gp.predict(NextPointValues.reshape(1, -1),return_std=True)
        #     target = f(**next_point_to_probe)
        #     optimizer.register(params=next_point_to_probe, target=target)
    else:

        # Bounded region of parameter space
        pbounds = {'x': (2, 4), 'y': (-3, 3)}

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

        optimizer.maximize(
            init_points=2,
            n_iter=3,
        )

