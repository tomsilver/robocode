#!/bin/bash
python -m black . --exclude 'prpl-mono|\.venv|outputs|multirun'
docformatter -i -r . --exclude venv .venv prpl-mono outputs multirun
isort . --skip .venv --skip prpl-mono --skip outputs --skip multirun
