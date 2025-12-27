.. toctree::
   :hidden:

   Quickstart <self>

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Example Notebooks:

   Basic Tour </basic-tour>
   Advanced Tour </advanced-tour>
   Constrained Bayesian Optimization </constraints>
   Parameter Types </parameter_types>
   Sequential Domain Reduction </domain_reduction>
   Acquisition Functions </acquisition_functions>
   Exploration vs. Exploitation </exploitation_vs_exploration>
   Visualization of a 1D-Optimization </visualization>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API reference:

   reference/bayes_opt
   reference/acquisition
   reference/constraint
   reference/domain_reduction
   reference/target_space
   reference/parameter
   reference/exception
   reference/other

.. raw:: html

   <div align="center">
     <img src="https://raw.githubusercontent.com/bayesian-optimization/BayesianOptimization/master/docsrc/static/func.png"><br><br>
   </div>

Bayesian Optimization
=====================

|tests| |Codecov| |Pypi| |PyPI - Python Version|

Pure Python implementation of bayesian global optimization with gaussian
processes.

This is a constrained global optimization package built upon bayesian
inference and gaussian processes, that attempts to find the maximum value
of an unknown function in as few iterations as possible. This technique
is particularly suited for optimization of high cost functions and
situations where the balance between exploration and exploitation is
important.

Installation
------------

pip (via PyPI)
~~~~~~~~~~~~~~

.. code:: console

   $ pip install bayesian-optimization

Conda (via conda-forge)
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: console

   $ conda install -c conda-forge bayesian-optimization

How does it work?
-----------------

Bayesian optimization works by constructing a posterior distribution of
functions (gaussian process) that best describes the function you want
to optimize. As the number of observations grows, the posterior
distribution improves, and the algorithm becomes more certain of which
regions in parameter space are worth exploring and which are not, as
seen in the picture below.

.. image:: ./static/bo_example.png
   :alt: BayesianOptimization in action

As you iterate over and over, the algorithm balances its needs of
exploration and exploitation taking into account what it knows about the
target function. At each step a Gaussian Process is fitted to the known
samples (points previously explored), and the posterior distribution,
combined with a exploration strategy (such as UCB (Upper Confidence
Bound), or EI (Expected Improvement)), are used to determine the next
point that should be explored (see the gif below).

.. image:: ./static/bayesian_optimization.gif
   :alt: BayesianOptimization in action

This process is designed to minimize the number of steps required to
find a combination of parameters that are close to the optimal
combination. To do so, this method uses a proxy optimization problem
(finding the maximum of the acquisition function) that, albeit still a
hard problem, is cheaper (in the computational sense) and common tools
can be employed. Therefore Bayesian Optimization is most adequate for
situations where sampling the function to be optimized is a very
expensive endeavor. See the references for a proper discussion of this
method.

This project is under active development, if you find a bug, or anything
that needs correction, please let us know by filing an
`issue on GitHub <https://github.com/bayesian-optimization/BayesianOptimization/issues>`__
.


Quick Index
-----------

See below for a quick tour over the basics of the Bayesian Optimization
package. More detailed information, other advanced features, and tips on
usage/implementation can be found in the
`examples <examples.html>`__
section. We suggest that you:

-  Follow the `basic tour
   notebook <basic-tour.html>`__
   to learn how to use the package's most important features.
-  Take a look at the `advanced tour
   notebook <advanced-tour.html>`__
   to learn how to make the package more flexible or how to use observers.
-  To learn more about acquisition functions, a central building block
   of bayesian optimization, see the `acquisition functions
   notebook <acquisition_functions.html>`__
-  If you want to optimize over integer-valued or categorical
   parameters, see the `parameter types
   notebook <parameter_types.html>`__.
-  Check out this
   `notebook <visualization.html>`__
   with a step by step visualization of how this method works.
-  To understand how to use bayesian optimization when additional
   constraints are present, see the `constrained optimization
   notebook <constraints.html>`__.
-  Explore the `domain reduction
   notebook <domain_reduction.html>`__
   to learn more about how search can be sped up by dynamically changing
   parameters' bounds.
-  Explore this
   `notebook <exploitation_vs_exploration.html>`__
   exemplifying the balance between exploration and exploitation and how
   to control it.
-  Go over this
   `script <https://github.com/bayesian-optimization/BayesianOptimization/blob/master/examples/sklearn_example.py>`__
   for examples of how to tune parameters of Machine Learning models
   using cross validation and bayesian optimization.
-  Finally, take a look at this
   `script <https://github.com/bayesian-optimization/BayesianOptimization/blob/master/examples/async_optimization.py>`__
   for ideas on how to implement bayesian optimization in a distributed
   fashion using this package.


Citation
--------

If you used this package in your research, please cite it:

::

   @Misc{,
       author={Fernando Nogueira},
       title={{Bayesian Optimization}: Open source constrained global optimization tool for {Python}},
       year={2014--},
       url="https://github.com/bayesian-optimization/BayesianOptimization"
   }

If you used any of the advanced functionalities, please additionally
cite the corresponding publication:

For the ``SequentialDomainTransformer``:

::

   @article{
       author={Stander, Nielen and Craig, Kenneth},
       year={2002},
       month={06},
       pages={},
       title={On the robustness of a simple domain reduction scheme for simulation-based optimization},
       volume={19},
       journal={International Journal for Computer-Aided Engineering and Software (Eng. Comput.)},
       doi={10.1108/02644400210430190}
   }

For constrained optimization:

::

   @inproceedings{gardner2014bayesian,
       title={Bayesian optimization with inequality constraints.},
       author={Gardner, Jacob R and Kusner, Matt J and Xu, Zhixiang Eddie and Weinberger, Kilian Q and Cunningham, John P},
       booktitle={ICML},
       volume={2014},
       pages={937--945},
       year={2014}
   }

For optimization over non-float parameters:

::

   @article{garrido2020dealing,
       title={Dealing with categorical and integer-valued variables in bayesian optimization with gaussian processes},
       author={Garrido-Merch{\'a}n, Eduardo C and Hern{\'a}ndez-Lobato, Daniel},
       journal={Neurocomputing},
       volume={380},
       pages={20--35},
       year={2020},
       publisher={Elsevier}
   }

.. |tests| image:: https://github.com/bayesian-optimization/BayesianOptimization/actions/workflows/run_tests.yml/badge.svg
.. |Codecov| image:: https://codecov.io/github/bayesian-optimization/BayesianOptimization/badge.svg?branch=master&service=github
   :target: https://codecov.io/github/bayesian-optimization/BayesianOptimization?branch=master
.. |Pypi| image:: https://img.shields.io/pypi/v/bayesian-optimization.svg
   :target: https://pypi.python.org/pypi/bayesian-optimization
.. |PyPI - Python Version| image:: https://img.shields.io/pypi/pyversions/bayesian-optimization

