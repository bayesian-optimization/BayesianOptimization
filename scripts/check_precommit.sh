#!/usr/bin/env sh
set -ex

poetry run pre-commit install
poetry run pre-commit run --all-files --show-diff-on-failure