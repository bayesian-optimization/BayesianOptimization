from setuptools import setup, find_packages

setup(
    name='bayesian-optimization',
    version='0.3.0',
    url='https://github.com/fmfn/BayesianOptimization',
    packages=find_packages(),
    description='Bayesian Optimization package',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "scikit-learn >= 0.18.0",
    ],
)
