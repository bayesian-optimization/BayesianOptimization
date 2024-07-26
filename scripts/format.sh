#!/usr/bin/env sh
set -ex

poetry run ruff format bayes_opt tests
poetry run ruff check bayes_opt tests --fix

