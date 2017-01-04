from setuptools import setup, find_packages

setup(
    name='bayesian-optimization',
    version='0.4.0',
    url='https://github.com/fmfn/BayesianOptimization',
    packages=find_packages(),
    author='Fernando Nogueira',
    author_email="fmfnogueira@gmail.com",
    description='Bayesian Optimization package',
    download_url = 'https://github.com/fmfn/BayesianOptimization/tarball/0.4',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "scikit-learn >= 0.18.0",
    ],
)
