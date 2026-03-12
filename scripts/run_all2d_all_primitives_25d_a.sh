#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiment.py -m \
    approach=agentic \
    approach.use_docker=true \
    approach.max_budget_usd=25 \
    seed=24 \
    'primitives=[check_action_collision,render_state,csp,BiRRT]' \
    environment=clutteredstorage2d_medium,stickbutton2d_medium,obstruction2d_medium,clutteredretrieval2d_medium \
    'hydra.sweep.dir=multirun/2026-03-10/all_primitives_25d_all2d_s24' \
    'hydra.sweep.subdir=s${seed}/${hydra:runtime.choices.environment}'
