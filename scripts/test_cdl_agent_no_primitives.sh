#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
ENV="${2:-obstruction2d_medium}"
DATE=$(date +%Y-%m-%d)

python experiments/run_experiment.py \
    approach=agentic_cdl \
    approach.use_docker=true \
    approach.max_budget_usd=20.0 \
    seed="$SEED" \
    num_eval_tasks=100 \
    'primitives=[]' \
    'mcp_tools=[]' \
    environment="$ENV" \
    approach.load_dir=outputs/cdl_no_primitives_w_helpers_${ENV}_${DATE}/s${SEED} \
    "hydra.run.dir=outputs/cdl_no_primitives_w_helpers_${ENV}_${DATE}/s${SEED}_test"
