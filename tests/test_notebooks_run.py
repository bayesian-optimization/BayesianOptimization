"""
collect all notebooks in examples, and check that they run without error
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

this_file_loc = Path(__file__).parent


def check_notebook_runs(notebook_loc: Path):
    print(f"running: {notebook_loc}...")
    try:
        with notebook_loc.open(encoding="utf8") as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": Path(notebook_loc).parent}})
    except Exception as exc:
        error_msg = f"failed to run notebook {notebook_loc}"
        raise Exception(error_msg) from exc  # noqa: TRY002 # FIXME
    print("success!")


def test_all_notebooks_run():
    # get all notebooks:
    notebooks_not_to_run = ["put_notebooks_to_skip_here"]
    for notebook in this_file_loc.with_name("examples").glob("*.ipynb"):
        as_string = str(notebook)
        if any([nb in as_string for nb in notebooks_not_to_run]):
            continue
        check_notebook_runs(notebook)
