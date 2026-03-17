#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
ENV="${2:-obstruction2d_medium}"

python experiments/run_experiment.py \
    approach=agentic_cdl \
    approach.use_docker=true \
    approach.max_budget_usd=1.0 \
    seed="$SEED" \
    'primitives=[]' \
    'mcp_tools=[render_state,render_policy]' \
    environment="$ENV" \
    "hydra.run.dir=outputs/cdl_no_primitives_${ENV}/s${SEED}"
