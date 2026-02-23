#!/usr/bin/env bash
# Run all easy 2D environments with all primitives in Docker (seeds 45-47).
set -euo pipefail

python experiments/run_experiment.py -m \
    approach=agentic \
    approach.use_docker=true \
    seed=424,222 \
    'primitives=[check_action_collision,render_state,csp,BiRRT]' \
    environment=motion2d_easy,obstruction2d_easy,clutteredretrieval2d_easy,clutteredstorage2d_easy,stickbutton2d_easy,pushpullhook2d \
    'hydra.sweep.dir=multirun/2026-02-23/all_primitives_5d_s45-47'
