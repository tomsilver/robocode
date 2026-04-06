#!/usr/bin/env bash
set -euo pipefail

ENV="${1:-obstruction2d_medium}"
DATE=$(date +%m-%d)

python experiments/run_experiment.py \
    approach=agentic_cdl \
    approach.use_docker=true \
    approach.max_budget_usd=20.0 \
    seed=42,24,444 \
    num_eval_tasks=100 \
    'primitives=[BiRRT]' \
    'mcp_tools=[render_state,render_policy]' \
    environment="$ENV" \
    "hydra.sweep.dir=outputs/cdl_mp_${ENV}_${DATE}" \
    "hydra.sweep.subdir=s${seed}"
