#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiment.py \
    approach=multi_session_agentic \
    approach.use_docker=true \
    approach.num_sessions=5 \
    'approach.session_budgets_usd=[5.0,5.0,5.0,5.0,5.0]' \
    seed=424 \
    'primitives=[]' \
    environment=clutteredstorage2d_medium,stickbutton2d_medium \
    'hydra.sweep.dir=multirun/2026-03-13/no_prim_multisession' \
    'hydra.sweep.subdir=s${seed}/${hydra:runtime.choices.environment}'