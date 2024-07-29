<div align="center">
  <img src="https://raw.githubusercontent.com/bayesian-optimization/BayesianOptimization/master/docsrc/static/func.png"><br><br>
</div>

# Bayesian Optimization

![tests](https://github.com/bayesian-optimization/BayesianOptimization/actions/workflows/run_tests.yml/badge.svg)
[![docs - stable](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbayesian-optimization%2FBayesianOptimization%2Fgh-pages%2Fversions.json&query=%24%5B%3F(%40.aliases%20%26%26%20%40.aliases.indexOf('stable')%20%3E%20-1)%5D.version&prefix=stable%20(v&suffix=)&label=docs)](https://bayesian-optimization.github.io/BayesianOptimization/)
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

This project is under active development, if you find a bug, or anything that
needs correction, please let me know.

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
