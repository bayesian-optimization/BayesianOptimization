from setuptools import setup, find_packages
import bayes_opt

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='bayesian-optimization',
    version=bayes_opt.__version__,
    url='https://github.com/fmfn/BayesianOptimization',
    packages=find_packages(),
    author='Fernando Nogueira',
    author_email="fmfnogueira@gmail.com",
    description='Bayesian Optimization package',
    long_description="A Python implementation of global optimization with gaussian processes.",
    long_description_content_type = "text/markdown",
    download_url='https://github.com/fmfn/BayesianOptimization/tarball/0.6',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 1.6.0",
        "scikit-learn >= 0.18.0",
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
    ]
)
