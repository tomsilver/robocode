#!/bin/bash
./run_autoformat.sh
mypy . --exclude prpl-mono --exclude outputs
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc --ignore=prpl-mono --ignore=outputs
pytest tests/
