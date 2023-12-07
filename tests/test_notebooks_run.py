"""
collect all notebooks in examples, and check that they run without error
"""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
from glob import glob
from matplotlib import pyplot as plt  # this script doesn't use this but the notebooks do
this_file_loc = Path(__file__).parent

def check_notebook_runs(notebook_loc):

    print(f'running: {notebook_loc}...')
    try:
        with open(notebook_loc, encoding='utf8') as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': Path(notebook_loc).parent}})
    except Exception as e:
        print(f'failed to run notebook {notebook_loc}: rethrowing exception:')
        raise e
    print(f'success!')

def test_all_notebooks_run():
    # get all notebooks:
    notebooks = glob(str(this_file_loc.parent / 'examples' / '*.ipynb'))
    notebooks_not_to_run = ['put_notebooks_to_skip_here']
    for notebook in notebooks:
        if any([nb in notebook for nb in notebooks_not_to_run]):
            continue
        check_notebook_runs(notebook)