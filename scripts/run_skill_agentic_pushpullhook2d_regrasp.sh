#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiment.py -m \
    approach=skill_agentic \
    approach.use_docker=true \
    approach.failed_state_dir=init_states/pushpullhook2d/regrasp \
    approach.success_state_dir=init_states/pushpullhook2d/successful \
    small_test=true \
    seed=42,24,424 \
    environment=pushpullhook2d \
    'hydra.sweep.dir=multirun/2026-03-15/skill_agentic_pushpullhook2d_regrasp' \
    'hydra.sweep.subdir=s${seed}/${hydra:runtime.choices.environment}'
