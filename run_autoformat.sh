#!/bin/bash
set -e
python -m black . --exclude 'third-party|\.venv|outputs|multirun|Results|ref/'
docformatter -i -r . --exclude venv .venv third-party outputs multirun Results ref
isort . --skip .venv --skip third-party --skip outputs --skip multirun --skip Results --skip ref
