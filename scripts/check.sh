#!/usr/bin/env sh
set -ex

poetry run ruff format --check bayes_opt tests
poetry run ruff check bayes_opt tests
