#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiment.py \
    approach=multi_session_agentic \
    approach.use_docker=true \
    approach.num_sessions=5 \
    'approach.session_budgets_usd=[5.0,5.0,5.0,5.0,5.0]' \
    seed=24 \
    'primitives=[]' \
    environment=stickbutton2d_medium
