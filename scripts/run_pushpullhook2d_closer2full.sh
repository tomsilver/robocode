#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiment.py -m \
    approach=progressive_agentic \
    approach.use_docker=true \
    seed=24,42,424 \
    approach.max_budget_usd=25 \
    'primitives=[check_action_collision,render_state,csp,BiRRT]' \
    environment=pushpullhook2d \
    'approach.resume_dir=${hydra:runtime.cwd}/multirun/pushpullhook2d_grasp_touch2closer_3seeds_np/s24/pushpullhook2d_grasp_touch_closer' \
    approach.resume_env=pushpullhook2d_grasp_touch_closer \
    'hydra.sweep.dir=multirun/pushpullhook2d_grasp_closer2full_3seeds' \
    'hydra.sweep.subdir=s${seed}/${hydra:runtime.choices.environment}'
