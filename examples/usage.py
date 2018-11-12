"""Example of how to use this bayesian optimization package."""

import sys
sys.path.append("./")
import time
from bayes_opt import BayesianOptimization


def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, however, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its outputs values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1


def main():
    "..."
    pbounds = {'x': (-4, 4), 'y': (-3, 3)}

    s = (
        "Lets find the maximum of a simple quadratic function of two " +
        "variables.\nWe create the bayes_opt object and pass the function " +
        "to be maximized together with the parameters names and their bounds."
    )
    print(s)
    bo = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=1,
        random_state=1,
    )

    s = (
        "The BayesianOptimization object will work all of the box without much " +
        "tuning needed."
    )
    bo.maximize(3, 2)

    s = (
        "The best combination of parameters and target value found can be " +
        "accessed via the property `bo.max`"
    )
    print(s)
    print(bo.max)

    s = (
        "While the list of parameters probes and their corresponding target " +
        "values is available via the property `bo.res`"
    )
    print(s)
    print(bo.res[:3])

    s = (
        "It is often the case that we have an idea of regions of the " +
        "parameter space where the maximum of our function might lie.\n" +
        "For these situations the BayesianOptimization object allows" +
        "the user to specify specific points to be probed.\n" +
        "These will be explored before the gaussian process takes over."
    )
    print(s)
    bo.probe(
        x={"x": 0, "y": 1},
        lazy=True
    )
    bo.probe(
        x=[1, 0],
        lazy=True
    )
    bo.maximize(init_points=0, n_iter=0)

    # Additionally, if we have any prior knowledge of the behaviour of
    # the target function (even if not totally accurate) we can also
    # tell that to the optimizer.
    # Here we pass a dictionary with 'target' and parameter names as keys and a
    # list of corresponding values
    # bo.initialize(
    #     {
    #         'target': [-1, -1],
    #         'x': [1, 1],
    #         'y': [0, 2]
    #     }
    # )

    # If we are not satisfied with the current results we can pickup from
    # where we left, maybe pass some more exploration points to the algorithm
    # change any parameters we may choose, and the let it run again.
    # bo.explore({'x': [0.6], 'y': [-0.23]})

    # Making changes to the gaussian process can impact the algorithm
    # dramatically.
    # gp_params = {'kernel': None,
    #              'alpha': 1e-5}

    # Run it again with different acquisition function
    # bo.maximize(n_iter=5, acq='ei', **gp_params)

    # Finally, we take a look at the final results.
    # print(bo.res['max'])
    # print(bo.res['all'])


if __name__ == "__main__":
    main()
