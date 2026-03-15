#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiment.py -m \
    approach=progressive_agentic \
    approach.use_docker=true \
    seed=424 \
    approach.max_budget_usd=25 \
    'primitives=[]' \
    environment=pushpullhook2d_grasp_touch_closer \
    'approach.resume_dir=${hydra:runtime.cwd}/multirun/pushpullhook2d_grasp_random2touch_3seeds_np/s${seed}/pushpullhook2d_grasp_touch' \
    approach.resume_env=pushpullhook2d_grasp_touch \
    'hydra.sweep.dir=multirun/pushpullhook2d_grasp_touch2closer_3seeds_np' \
    'hydra.sweep.subdir=s${seed}/${hydra:runtime.choices.environment}'
