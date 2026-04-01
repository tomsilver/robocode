#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
ENV="${2:-obstruction2d_medium}"
DATE=$(date +%m-%d)

python experiments/run_experiment.py \
    approach=agentic_cdl \
    approach.use_docker=true \
    approach.max_budget_usd=20.0 \
    seed="$SEED" \
    num_eval_tasks=100 \
    'primitives=[BiRRT]' \
    'mcp_tools=[render_state,render_policy]' \
    environment="$ENV" \
    "hydra.run.dir=outputs/cdl_mp_${ENV}_${DATE}/s${SEED}"
