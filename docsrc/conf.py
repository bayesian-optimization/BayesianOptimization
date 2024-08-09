# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import time
import shutil
from glob import glob
from pathlib import Path
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

# copy the latest example files:
this_file_loc = Path(__file__).parent
notebooks = glob(str(this_file_loc.parent / 'examples' / '*.ipynb'))
for notebook in notebooks:
    shutil.copy(notebook, this_file_loc)


# -- Project information -----------------------------------------------------

project = 'bayesian-optimization'
author = 'Fernando Nogueira'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.githubpages',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.mathjax',
    "sphinx.ext.napoleon",
    'sphinx.ext.intersphinx',
    'sphinx_immaterial',
]

source_suffix = {
    '.rst': 'restructuredtext',
}
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Link types to the corresponding documentations
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}


napoleon_use_rtype = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_title = "Bayesian Optimization"
html_theme = "sphinx_immaterial"
copyright = f"{time.strftime('%Y')}, Fernando Nogueira and the bayesian-optimization developers"

# material theme options (see theme.conf for more information)
html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-edit-outline",
    },
    "site_url": "https://bayesian-optimization.github.io/BayesianOptimization/",
    "repo_url": "https://github.com/bayesian-optimization/BayesianOptimization/",
    "repo_name": "bayesian-optimization",
    "edit_uri": "blob/master/docsrc",
    "globaltoc_collapse": True,
    "features": [
        "navigation.expand",
        # "navigation.tabs",
        # "toc.integrate",
        "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        "navigation.top",
        # "navigation.tracking",
        # "search.highlight",
        "search.share",
        "toc.follow",
        "toc.sticky",
        "content.tabs.link",
        "announce.dismiss",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "light-blue",
            "accent": "light-green",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "deep-orange",
            "accent": "lime",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to light mode",
            },
        },
    ],
    # BEGIN: version_dropdown
    "version_dropdown": True,
    "version_json": '../versions.json',
    # END: version_dropdown
    "scope": "/", # share preferences across subsites
    "toc_title_is_page_title": True,
    # BEGIN: social icons
    "social": [
        {
            "icon": "fontawesome/brands/github",
            "link": "https://github.com/bayesian-optimization/BayesianOptimization",
            "name": "Source on github.com",
        },
        {
            "icon": "fontawesome/brands/python",
            "link": "https://pypi.org/project/bayesian-optimization/",
        },
        {
            "icon": "fontawesome/brands/python",
            "link": "https://anaconda.org/conda-forge/bayesian-optimization",
        }
    ],
    # END: social icons
}

html_favicon = 'func.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']