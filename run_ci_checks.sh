#!/bin/bash
./run_autoformat.sh
mypy . --exclude prpl-mono
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc --ignore=prpl-mono
pytest tests/
