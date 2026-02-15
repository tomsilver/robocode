#!/bin/bash
python -m black . --exclude 'prpl-mono|\.venv|outputs'
docformatter -i -r . --exclude venv .venv prpl-mono outputs
isort . --skip .venv --skip prpl-mono --skip outputs
