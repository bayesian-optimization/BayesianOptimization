import numpy as np
from bayes_opt import BayesianOptimization, ConstraintModel
from pytest import approx, raises

np.random.seed(42)


def test_single_constraint():

    def target_function(x, y):
        return np.cos(2 * x) * np.cos(y) + np.sin(x)

    def constraint_function(x, y):
        return np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y)

    constraint_limit = 0.5

    conmod = ConstraintModel(constraint_function, constraint_limit)
    pbounds = {'x': (0, 6), 'y': (0, 6)}

    optimizer = BayesianOptimization(
        f=target_function,
        constraint=conmod,
        pbounds=pbounds,
        verbose=0,
        random_state=1,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=10,
    )


def test_single_constraint_max_is_allowed():

    def target_function(x, y):
        return np.cos(2 * x) * np.cos(y) + np.sin(x)

    def constraint_function(x, y):
        return np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y)

    constraint_limit = 0.5

    conmod = ConstraintModel(constraint_function, constraint_limit)
    pbounds = {'x': (0, 6), 'y': (0, 6)}

    optimizer = BayesianOptimization(
        f=target_function,
        constraint=conmod,
        pbounds=pbounds,
        verbose=0,
        random_state=1,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=10,
    )

    assert constraint_function(**optimizer.max["params"]) <= constraint_limit


def test_accurate_approximation_when_known():

    def target_function(x, y):
        return np.cos(2 * x) * np.cos(y) + np.sin(x)

    def constraint_function(x, y):
        return np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y)

    constraint_limit = 0.5

    conmod = ConstraintModel(constraint_function, constraint_limit)
    pbounds = {'x': (0, 6), 'y': (0, 6)}

    optimizer = BayesianOptimization(
        f=target_function,
        constraint=conmod,
        pbounds=pbounds,
        verbose=0,
        random_state=1,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=10,
    )

    # Exclude the last sampled point, because the constraint is not fitted on that.
    res = np.array([[r['target'], r['constraint'], r['params']['x'], r['params']['y']] for r in optimizer.res[:-1]])

    xy = res[:, [2, 3]]
    x = res[:, 2]
    y = res[:, 3]
    
    assert constraint_function(x, y) == approx(conmod.approx(xy), rel=1e-5, abs=1e-5)
    assert constraint_function(x, y) == approx(optimizer.space.constraint_values[:-1], rel=1e-5, abs=1e-5)


def test_multiple_constraints():

    def target_function(x, y):
        return np.cos(2 * x) * np.cos(y) + np.sin(x)

    def constraint_function_2_dim(x, y):
        return np.array([
            -np.cos(x) * np.cos(y) + np.sin(x) * np.sin(y),
            -np.cos(x) * np.cos(-y) + np.sin(x) * np.sin(-y)
        ])

    constraint_limit = np.array([0.6, 0.6])

    conmod = ConstraintModel(constraint_function_2_dim, constraint_limit)
    pbounds = {'x': (0, 6), 'y': (0, 6)}

    optimizer = BayesianOptimization(
        f=target_function,
        constraint=conmod,
        pbounds=pbounds,
        verbose=0,
        random_state=1,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=10,
    )

    assert np.all(
        constraint_function_2_dim(
            **optimizer.max["params"]) <= constraint_limit)
    
    params = optimizer.res[0]["params"]
    x, y = params['x'], params['y']

    assert constraint_function_2_dim(x, y) == approx(conmod.approx(np.array([x, y])), rel=1e-5, abs=1e-5)


def test_kwargs_not_the_same():

    def target_function(x, y):
        return np.cos(2 * x) * np.cos(y) + np.sin(x)

    def constraint_function(a, b):
        return np.cos(a) * np.cos(b) - np.sin(a) * np.sin(b)

    constraint_limit = 0.5

    conmod = ConstraintModel(constraint_function, constraint_limit)
    pbounds = {'x': (0, 6), 'y': (0, 6)}

    optimizer = BayesianOptimization(
        f=target_function,
        constraint=conmod,
        pbounds=pbounds,
        verbose=0,
        random_state=1,
    )
    with raises(TypeError, match="Encountered TypeError when evaluating"):
        optimizer.maximize(
            init_points=2,
            n_iter=10,
        )