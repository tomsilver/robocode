#!/bin/bash
./run_autoformat.sh
mypy . --exclude prpl-mono --exclude outputs --exclude multirun
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc --ignore=prpl-mono --ignore=outputs --ignore=multirun
pytest tests/
