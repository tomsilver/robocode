#!/bin/bash
set -e
./run_autoformat.sh
mypy . --exclude prpl-mono --exclude outputs --exclude multirun --exclude 'src/robocode/mcp' --exclude 'tests/mcp'
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc --ignore=prpl-mono --ignore=outputs --ignore=multirun --ignore=tests/mcp --ignore=src/robocode/mcp
pytest tests/ --ignore=tests/mcp
