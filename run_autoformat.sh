#!/bin/bash
python -m black . --exclude 'prpl-mono|\.venv'
docformatter -i -r . --exclude venv .venv prpl-mono
isort . --skip .venv --skip prpl-mono
