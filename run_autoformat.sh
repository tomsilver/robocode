#!/bin/bash
python -m black . --exclude 'third-party|\.venv|outputs|multirun'
docformatter -i -r . --exclude venv .venv third-party outputs multirun
isort . --skip .venv --skip third-party --skip outputs --skip multirun
