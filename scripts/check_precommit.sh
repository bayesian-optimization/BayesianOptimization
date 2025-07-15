#!/usr/bin/env sh
set -ex

uv run pre-commit install
uv run pre-commit run --all-files --show-diff-on-failure
