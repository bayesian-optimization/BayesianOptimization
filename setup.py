from setuptools import setup, find_packages

setup(
    name='bayesian-optimization',
    version='1.2.0',
    url='https://github.com/fmfn/BayesianOptimization',
    packages=find_packages(),
    author='Fernando Nogueira',
    author_email="fmfnogueira@gmail.com",
    description='Bayesian Optimization package',
    long_description='A Python implementation of global optimization with gaussian processes.',
    download_url='https://github.com/fmfn/BayesianOptimization/tarball/0.6',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "scikit-learn >= 0.18.0",
    ],
)
