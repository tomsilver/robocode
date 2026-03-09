#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiment.py -m \
    approach=agentic \
    approach.use_docker=true \
    seed=42,24,424 \
    approach.max_budget_usd=25 \
    'primitives=[check_action_collision,render_state,csp,BiRRT]' \
    environment=pushpullhook2d_grasp_touch \
    'hydra.sweep.dir=multirun/pushpullhook2d_grasp_touch_3seeds' \
    'hydra.sweep.subdir=s${seed}/${hydra:runtime.choices.environment}'
