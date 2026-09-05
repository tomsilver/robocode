#!/bin/bash
set -e
./run_autoformat.sh
mypy . --exclude third-party --exclude outputs --exclude multirun --exclude Results --exclude ref/ --exclude 'src/robocode/mcp'
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc --ignore=third-party --ignore=outputs --ignore=multirun --ignore=Results --ignore=ref
pytest tests/ --ignore=tests/mcp
