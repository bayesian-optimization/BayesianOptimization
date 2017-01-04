# Bayesian Optimization

Pure Python implementation of bayesian global optimization with gaussian
processes.

    pip install bayesian-optimization

This is a constrained global optimization package built upon bayesian inference
and gaussian process, that attempts to find the maximum value of an unknown
function in as few iterations as possible. This technique is particularly
suited for optimization of high cost functions, situations where the balance
between exploration and exploitation is important.

## Quick Start
In the [examples](https://github.com/fmfn/BayesianOptimization/tree/master/examples)
folder you can get a grip of how the method and this package work by:
- Checking out this
[notebook](https://github.com/fmfn/BayesianOptimization/blob/master/examples/visualization.ipynb)
with a step by step visualization of how this method works.
- Going over this
[script](https://github.com/fmfn/BayesianOptimization/blob/master/examples/usage.py)
to become familiar with this packages basic functionalities.
- Exploring this [notebook](https://github.com/fmfn/BayesianOptimization/blob/master/examples/exploitation%20vs%20exploration.ipynb)
exemplifying the balance between exploration and exploitation and how to
control it.
- Checking out these scripts ([sklearn](https://github.com/fmfn/BayesianOptimization/blob/master/examples/sklearn_example.py),
[xgboost](https://github.com/fmfn/BayesianOptimization/blob/master/examples/xgboost_example.py))
for examples of how to use this package to tune parameters of ML estimators
using cross validation and bayesian optimization.


## How does it work?

Bayesian optimization works by constructing a posterior distribution of functions (gaussian process) that best describes the function you want to optimize. As the number of observations grows, the posterior distribution improves, and the algorithm becomes more certain of which regions in parameter space are worth exploring and which are not, as seen in the picture below.

![BayesianOptimization in action](https://github.com/fmfn/BayesianOptimization/blob/master/examples/bo_example.png)

As you iterate over and over, the algorithm balances its needs of exploration and exploitation taking into account what it knows about the target function. At each step a Gaussian Process is fitted to the known samples (points previously explored), and the posterior distribution, combined with a exploration strategy (such as UCB (Upper Confidence Bound), or EI (Expected Improvement)), are used to determine the next point that should be explored (see the gif below).

![BayesianOptimization in action](https://github.com/fmfn/BayesianOptimization/blob/master/examples/bayesian_optimization.gif)

This process is designed to minimize the number of steps required to find a combination of parameters that are close to the optimal combination. To do so, this method uses a proxy optimization problem (finding the maximum of the acquisition function) that, albeit still a hard problem, is cheaper (in the computational sense) and common tools can be employed. Therefore Bayesian Optimization is most adequate for situations where sampling the function to be optimized is a very expensive endeavor. See the references for a proper discussion of this method.

This project is under active development, if you find a bug, or anything that
needs correction, please let me know.

Installation
============

### Installation

For the latest release, run:

    pip install bayesian-optimization

The bleeding edge version can be installed with:

    pip install git+https://github.com/fmfn/BayesianOptimization.git

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from Github and install all dependencies:

    git clone https://github.com/fmfn/BayesianOptimization.git
    cd BayesianOptimization
    python setup.py install

### Dependencies
* Numpy
* Scipy
* Scikit-learn

### References:
* http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
* http://arxiv.org/pdf/1012.2599v1.pdf
* http://www.gaussianprocess.org/gpml/
* https://www.youtube.com/watch?v=vz3D36VXefI&index=10&list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6
