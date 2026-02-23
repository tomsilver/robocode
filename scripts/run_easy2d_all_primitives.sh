#!/usr/bin/env bash
# Run all easy 2D environments with no primitives in Docker.
set -euo pipefail

python experiments/run_experiment.py -m \
    approach=agentic \
    approach.use_docker=true \
    'primitives=[check_action_collision,render_state,csp,BiRRT]' \
    environment=motion2d_easy,obstruction2d_easy,clutteredretrieval2d_easy,clutteredstorage2d_easy,stickbutton2d_easy,pushpullhook2d
