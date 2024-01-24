import pytest
import numpy as np

from bayes_opt import BayesianOptimization
from bayes_opt.util import  Colours
from bayes_opt.util import load_logs, ensure_rng

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from scipy.optimize import NonlinearConstraint
from pathlib import Path
test_dir = Path(__file__).parent.resolve()


def test_logs():
    import pytest
    def f(x, y):
        return -x ** 2 - (y - 1) ** 2 + 1

    optimizer = BayesianOptimization(
        f=f,
        pbounds={"x": (-2, 2), "y": (-2, 2)}
    )
    assert len(optimizer.space) == 0

    load_logs(optimizer, [str(test_dir / "test_logs.log")])
    assert len(optimizer.space) == 5

    load_logs(optimizer, [str(test_dir / "test_logs.log")])
    assert len(optimizer.space) == 5

    other_optimizer = BayesianOptimization(
        f=lambda x: -x ** 2,
        pbounds={"x": (-2, 2)}
    )
    with pytest.raises(ValueError):
        load_logs(other_optimizer, [str(test_dir / "test_logs.log")])


def test_logs_constraint():

    def f(x, y):
        return -x ** 2 - (y - 1) ** 2 + 1

    def c(x, y):
        return x ** 2 + y ** 2
    
    constraint = NonlinearConstraint(c, -np.inf, 3)

    optimizer = BayesianOptimization(
        f=f,
        pbounds={"x": (-2, 2), "y": (-2, 2)},
        constraint=constraint
    )

    with pytest.raises(KeyError):
        load_logs(optimizer, [str(test_dir / "test_logs.log")])
    
    load_logs(optimizer, [str(test_dir / "test_logs_constrained.log")])

    assert len(optimizer.space) == 7


def test_colours():
    colour_wrappers = [
        (Colours.END, Colours.black),
        (Colours.BLUE, Colours.blue),
        (Colours.BOLD, Colours.bold),
        (Colours.CYAN, Colours.cyan),
        (Colours.DARKCYAN, Colours.darkcyan),
        (Colours.GREEN, Colours.green),
        (Colours.PURPLE, Colours.purple),
        (Colours.RED, Colours.red),
        (Colours.UNDERLINE, Colours.underline),
        (Colours.YELLOW, Colours.yellow),
    ]

    for colour, wrapper in colour_wrappers:
        text1 = Colours._wrap_colour("test", colour)
        text2 = wrapper("test")

        assert text1.split("test") == [colour, Colours.END]
        assert text2.split("test") == [colour, Colours.END]


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_target_space.py
    """
    import pytest
    pytest.main([__file__])
