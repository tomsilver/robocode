#!/bin/bash
set -e
git submodule update --init --recursive
uv sync --all-extras --dev
