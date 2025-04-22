"""
collect all notebooks in examples, and check that they run without error
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient.exceptions import CellExecutionError
from nbconvert.preprocessors import ExecutePreprocessor

this_file_loc = Path(__file__).parent
_NOTEBOOKS_NOT_TO_RUN = frozenset(["put_notebooks_to_skip_here"])


# get all notebooks:
@pytest.mark.parametrize("notebook", this_file_loc.with_name("examples").glob("*.ipynb"))
def test_all_notebooks_run(notebook: Path):
    as_string = str(notebook)
    if any([nb in as_string for nb in _NOTEBOOKS_NOT_TO_RUN]):
        pytest.skip(f"skipping [{notebook!s}]")

    print(f"running: {notebook}...")
    with notebook.open(encoding="utf8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": notebook.parent}})
        print("success!")
    except CellExecutionError as e:
        # Wrap the original error with the notebook name
        pytest.fail(f"Error executing notebook {notebook}: {e!s}")
