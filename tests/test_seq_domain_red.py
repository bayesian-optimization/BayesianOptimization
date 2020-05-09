from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt import BayesianOptimization


def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1


def test_bound_x_maximize():

    class Tracker:
        def __init__(self):
            self.start_count = 0
            self.step_count = 0
            self.end_count = 0

        def update_start(self, event, instance):
            self.start_count += 1

        def update_step(self, event, instance):
            self.step_count += 1

        def update_end(self, event, instance):
            self.end_count += 1

        def reset(self):
            self.__init__()

    bounds_transformer = SequentialDomainReductionTransformer()
    pbounds = {'x': (-10, 10), 'y': (-10, 10)}
    n_iter = 10

    standard_optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )

    standard_optimizer.maximize(
        init_points=2,
        n_iter=n_iter,
    )

    mutated_optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
        bounds_transformer=bounds_transformer
    )

    mutated_optimizer.maximize(
        init_points=2,
        n_iter=n_iter,
    )

    assert len(standard_optimizer.space) == len(mutated_optimizer.space)
    assert not (standard_optimizer._space.bounds ==
                mutated_optimizer._space.bounds).any()
