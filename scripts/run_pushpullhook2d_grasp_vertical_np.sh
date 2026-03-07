#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiment.py -m \
    approach=agentic \
    approach.use_docker=true \
    approach.max_budget_usd=25 \
    seed=42,24,424 \
    'primitives=[]' \
    environment=pushpullhook2d_grasp_vertical \
    'hydra.sweep.dir=multirun/pushpullhook2d_grasp_vertical_3seeds_np' \
    'hydra.sweep.subdir=s${seed}/${hydra:runtime.choices.environment}'
