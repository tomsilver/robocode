#!/usr/bin/env bash
set -euo pipefail

SEED=24
ENV=obstruction2d_medium

python experiments/run_experiment.py \
    approach=agentic_cdl \
    approach.use_docker=true \
    approach.max_budget_usd=20.0 \
    environment=stickbutton2d_medium \
    record_approach_history=true \
    seed="$SEED" \
    approach.load_dir=outputs/cdl_no_primitives_${ENV}/s${SEED} \
    'primitives=[]' \
    'mcp_tools=[]' \
    environment="$ENV" \
    "hydra.run.dir=outputs/cdl_no_primitives_visualize_${ENV}/s${SEED}"