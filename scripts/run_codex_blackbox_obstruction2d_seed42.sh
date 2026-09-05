#!/usr/bin/env bash
set -euo pipefail
: "${EVAL_SEED:?Set EVAL_SEED to the private evaluation-suite seed}"

python experiments/run_experiment.py \
    approach=agentic \
    approach/backend=codex_gpt6 \
    approach.container_backend=docker \
    approach.blackbox=true \
    approach.max_budget_usd=20.0 \
    environment=obstruction2d_generalized \
    primitive_level=none \
    replicate_seed=42 \
    eval_seed="$EVAL_SEED" \
    hydra.run.dir=outputs/codex_validation/blackbox_obstruction2d_gpt6_20usd
