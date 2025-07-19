#!/usr/bin/env sh
set -ex

uv run ruff format --check bayes_opt tests
uv run ruff check bayes_opt tests
