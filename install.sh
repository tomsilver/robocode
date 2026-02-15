#!/bin/bash
set -e
git submodule update --init --recursive
uv pip install \
    -e prpl-mono/relational-structs \
    -e prpl-mono/prpl-utils \
    -e prpl-mono/toms-geoms-2d \
    -e prpl-mono/kinder \
    -e ".[develop]"
