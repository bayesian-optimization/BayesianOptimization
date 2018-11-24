# import numpy as np

# from bayes_opt import BayesianOptimization
# from bayes_opt.util import ensure_rng


# def test_simple_optimization():
#     """
#     ...
#     """
#     def f(x, y):
#         return -x ** 2 - (y - 1) ** 2 + 1


#     optimizer = BayesianOptimization(
#         f=f,
#         pbounds={"x": (-3, 3), "y": (-3, 3)},
#         random_state=12356,
#         verbose=0,
#     )

#     optimizer.maximize(init_points=0, n_iter=25)

#     max_target = optimizer.max["target"]
#     max_x = optimizer.max["params"]["x"]
#     max_y = optimizer.max["params"]["y"]

#     assert (1 - max_target) < 1e-3
#     assert np.abs(max_x - 0) < 1e-1
#     assert np.abs(max_y - 1) < 1e-1


# def test_intermediate_optimization():
#     """
#     ...
#     """
#     def f(x, y, z):
#         x_factor = np.exp(-(x - 2) ** 2) + (1 / (x ** 2 + 1))
#         y_factor = np.exp(-(y - 6) ** 2 / 10)
#         z_factor = (1 + 0.2 * np.cos(z)) / (1 + z ** 2)
#         return (x_factor + y_factor) * z_factor

#     optimizer = BayesianOptimization(
#         f=f,
#         pbounds={"x": (-7, 7), "y": (-7, 7), "z": (-7, 7)},
#         random_state=56,
#         verbose=0,
#     )

#     optimizer.maximize(init_points=0, n_iter=150)

#     max_target = optimizer.max["target"]
#     max_x = optimizer.max["params"]["x"]
#     max_y = optimizer.max["params"]["y"]
#     max_z = optimizer.max["params"]["z"]

#     assert (2.640 - max_target) < 0
#     assert np.abs(2 - max_x) < 1e-1
#     assert np.abs(6 - max_y) < 1e-1
#     assert np.abs(0 - max_z) < 1e-1


# if __name__ == '__main__':
#     r"""
#     CommandLine:
#         python tests/test_bayesian_optimization.py
#     """
#     import pytest
#     pytest.main([__file__])
