#!/usr/bin/env bash
set -euo pipefail
: "${EVAL_SEED:?Set EVAL_SEED to the private evaluation-suite seed}"

REPLICATE_SEED=24
ENV=obstruction2d_medium

python experiments/run_experiment.py \
    approach=agentic_cdl \
    approach.container_backend=docker \
    approach.max_budget_usd=20.0 \
    environment=stickbutton2d_medium \
    record_approach_history=true \
    replicate_seed="$REPLICATE_SEED" \
    eval_seed="$EVAL_SEED" \
    approach.load_dir=outputs/cdl_no_primitives_${ENV}/s${REPLICATE_SEED} \
    'primitives=[]' \
    'mcp_tools=[]' \
    environment="$ENV" \
    "hydra.run.dir=outputs/cdl_no_primitives_visualize_${ENV}/s${REPLICATE_SEED}"
