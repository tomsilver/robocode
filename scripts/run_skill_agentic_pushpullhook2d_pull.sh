#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiment.py -m \
    approach=skill_agentic \
    approach.use_docker=true \
    approach.max_budget_usd=15.0 \
    approach.failed_state_dir=init_states/pushpullhook2d/pull \
    approach.success_state_dir=init_states/pushpullhook2d/successful \
    small_test=true \
    seed=42,24,424 \
    environment=pushpullhook2d \
    'hydra.sweep.dir=multirun/2026-03-17/skill_agentic_pushpullhook2d_pull' \
    'hydra.sweep.subdir=s${seed}/${hydra:runtime.choices.environment}'
