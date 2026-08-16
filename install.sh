#!/bin/bash
set -e
git submodule update --init --recursive

# `--no-extra libero` keeps the heavy, Linux-only LIBERO-PRO stack out of the
# default install; see the LIBERO-PRO section of the README to opt in. The
# Python version comes from .python-version (3.11).
uv sync --all-extras --no-extra libero --dev
