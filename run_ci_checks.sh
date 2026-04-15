#!/bin/bash
set -e
./run_autoformat.sh
mypy . --exclude prpl-mono --exclude outputs --exclude multirun --exclude analysis --exclude 'src/robocode/mcp'
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc --ignore=prpl-mono --ignore=outputs --ignore=multirun --ignore=analysis
pytest tests/ --ignore=tests/mcp
