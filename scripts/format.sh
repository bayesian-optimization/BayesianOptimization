#!/usr/bin/env sh
set -ex

uv run ruff format bayes_opt tests
uv run ruff check bayes_opt --fix
