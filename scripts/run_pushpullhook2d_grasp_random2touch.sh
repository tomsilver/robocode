#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiment.py -m \
    approach=progressive_agentic \
    approach.use_docker=true \
    seed=42,24,424 \
    approach.max_budget_usd=25 \
    'primitives=[check_action_collision,render_state,csp,BiRRT]' \
    environment=pushpullhook2d_grasp_touch \
    'approach.resume_dir=${hydra:runtime.cwd}/multirun/pushpullhook2d_grasp_random_3seeds/s${seed}/pushpullhook2d_grasp_random' \
    approach.resume_env=pushpullhook2d_grasp_random \
    'hydra.sweep.dir=multirun/pushpullhook2d_grasp_random2touch_3seeds' \
    'hydra.sweep.subdir=s${seed}/${hydra:runtime.choices.environment}'
