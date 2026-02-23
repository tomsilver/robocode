#!/usr/bin/env bash
# Run all easy 2D environments with all primitives ($25 budget) in Docker (seeds 43-44).
set -euo pipefail

python experiments/run_experiment.py -m \
    approach=agentic \
    approach.use_docker=true \
    approach.max_budget_usd=25 \
    seed=24,444 \
    'primitives=[check_action_collision,render_state,csp,BiRRT]' \
    environment=motion2d_easy,obstruction2d_easy,clutteredretrieval2d_easy,clutteredstorage2d_easy,stickbutton2d_easy,pushpullhook2d \
    'hydra.sweep.dir=multirun/2026-02-23/all_primitives_25d_s43-44'
