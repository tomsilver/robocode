#!/bin/bash
set -e
git submodule update --init --recursive
uv pip install -e prpl-mono/prpl-utils -e ".[develop]"
