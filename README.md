<div align="center">
  <img src="https://raw.githubusercontent.com/bayesian-optimization/BayesianOptimization/master/docsrc/static/func.png"><br><br>
</div>

# Bayesian Optimization

![tests](https://github.com/bayesian-optimization/BayesianOptimization/actions/workflows/run_tests.yml/badge.svg)
[![docs - stable](https://img.shields.io/badge/docs-stable-blue)](https://bayesian-optimization.github.io/BayesianOptimization/index.html)
[![Codecov](https://codecov.io/github/bayesian-optimization/BayesianOptimization/badge.svg?branch=master&service=github)](https://codecov.io/github/bayesian-optimization/BayesianOptimization?branch=master)
[![Pypi](https://img.shields.io/pypi/v/bayesian-optimization.svg)](https://pypi.python.org/pypi/bayesian-optimization)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bayesian-optimization)


Pure Python implementation of bayesian global optimization with gaussian
processes.


This is a constrained global optimization package built upon bayesian inference
and gaussian processes, that attempts to find the maximum value of an unknown
function in as few iterations as possible. This technique is particularly
suited for optimization of high cost functions and situations where the balance
between exploration and exploitation is important.

## Installation

* pip (via PyPI):

```console
$ pip install bayesian-optimization
```

* Conda (via conda-forge):

```console
$ conda install -c conda-forge bayesian-optimization
```

## How does it work?

See the [documentation](https://bayesian-optimization.github.io/BayesianOptimization/) for how to use this package.

Bayesian optimization works by constructing a posterior distribution of functions (gaussian process) that best describes the function you want to optimize. As the number of observations grows, the posterior distribution improves, and the algorithm becomes more certain of which regions in parameter space are worth exploring and which are not, as seen in the picture below.

![BayesianOptimization in action](docsrc/static/bo_example.png)

As you iterate over and over, the algorithm balances its needs of exploration and exploitation taking into account what it knows about the target function. At each step a Gaussian Process is fitted to the known samples (points previously explored), and the posterior distribution, combined with a exploration strategy (such as UCB (Upper Confidence Bound), or EI (Expected Improvement)), are used to determine the next point that should be explored (see the gif below).

![BayesianOptimization in action](docsrc/static/bayesian_optimization.gif)

This process is designed to minimize the number of steps required to find a combination of parameters that are close to the optimal combination. To do so, this method uses a proxy optimization problem (finding the maximum of the acquisition function) that, albeit still a hard problem, is cheaper (in the computational sense) and common tools can be employed. Therefore Bayesian Optimization is most adequate for situations where sampling the function to be optimized is a very expensive endeavor. See the references for a proper discussion of this method.

This project is under active development. If you run into trouble, find a bug or notice
anything that needs correction, please let us know by filing an issue.


## Basic tour of the Bayesian Optimization package

### 1. Specifying the function to be optimized

This is a function optimization package, therefore the first and most important ingredient is, of course, the function to be optimized.

**DISCLAIMER:** We know exactly how the output of the function below depends on its parameter. Obviously this is just an example, and you shouldn't expect to know it in a real scenario. However, it should be clear that you don't need to. All you need in order to use this package (and more generally, this technique) is a function `f` that takes a known set of parameters and outputs a real number.


```python
def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1
```

### 2. Getting Started

All we need to get started is to instantiate a `BayesianOptimization` object specifying a function to be optimized `f`, and its parameters with their corresponding bounds, `pbounds`. This is a constrained optimization technique, so you must specify the minimum and maximum values that can be probed for each parameter in order for it to work


```python
from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)
```

The BayesianOptimization object will work out of the box without much tuning needed. The main method you should be aware of is `maximize`, which does exactly what you think it does.

There are many parameters you can pass to maximize, nonetheless, the most important ones are:
- `n_iter`: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
- `init_points`: How many steps of **random** exploration you want to perform. Random exploration can help by diversifying the exploration space.


```python
optimizer.maximize(
    init_points=2,
    n_iter=3,
)
```

    |   iter    |  target   |     x     |     y     |
    -------------------------------------------------
    |  1        | -7.135    |  2.834    |  1.322    |
    |  2        | -7.78     |  2.0      | -1.186    |
    |  3        | -19.0     |  4.0      |  3.0      |
    |  4        | -16.3     |  2.378    | -2.413    |
    |  5        | -4.441    |  2.105    | -0.005822 |
    =================================================


The best combination of parameters and target value found can be accessed via the property `optimizer.max`.


```python
print(optimizer.max)
>>> {'target': -4.441293113411222, 'params': {'y': -0.005822117636089974, 'x': 2.104665051994087}}
```


While the list of all parameters probed and their corresponding target values is available via the property `optimizer.res`.


```python
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

>>> Iteration 0:
>>>     {'target': -7.135455292718879, 'params': {'y': 1.3219469606529488, 'x': 2.8340440094051482}}
>>> Iteration 1:
>>>     {'target': -7.779531005607566, 'params': {'y': -1.1860045642089614, 'x': 2.0002287496346898}}
>>> Iteration 2:
>>>     {'target': -19.0, 'params': {'y': 3.0, 'x': 4.0}}
>>> Iteration 3:
>>>     {'target': -16.29839645063864, 'params': {'y': -2.412527795983739, 'x': 2.3776144540856503}}
>>> Iteration 4:
>>>     {'target': -4.441293113411222, 'params': {'y': -0.005822117636089974, 'x': 2.104665051994087}}
```


## Minutiae

### Citation

If you used this package in your research, please cite it:

```
@Misc{,
    author = {Fernando Nogueira},
    title = {{Bayesian Optimization}: Open source constrained global optimization tool for {Python}},
    year = {2014--},
    url = " https://github.com/bayesian-optimization/BayesianOptimization"
}
```
If you used any of the advanced functionalities, please additionally cite the corresponding publication:

For the `SequentialDomainTransformer`:
```
@article{
    author = {Stander, Nielen and Craig, Kenneth},
    year = {2002},
    month = {06},
    pages = {},
    title = {On the robustness of a simple domain reduction scheme for simulation-based optimization},
    volume = {19},
    journal = {International Journal for Computer-Aided Engineering and Software (Eng. Comput.)},
    doi = {10.1108/02644400210430190}
}
```

For constrained optimization:
```
@inproceedings{gardner2014bayesian,
    title={Bayesian optimization with inequality constraints.},
    author={Gardner, Jacob R and Kusner, Matt J and Xu, Zhixiang Eddie and Weinberger, Kilian Q and Cunningham, John P},
    booktitle={ICML},
    volume={2014},
    pages={937--945},
    year={2014}
}
```

For optimization over non-float parameters:
```
@article{garrido2020dealing,
  title={Dealing with categorical and integer-valued variables in bayesian optimization with gaussian processes},
  author={Garrido-Merch{\'a}n, Eduardo C and Hern{\'a}ndez-Lobato, Daniel},
  journal={Neurocomputing},
  volume={380},
  pages={20--35},
  year={2020},
  publisher={Elsevier}
}
```
