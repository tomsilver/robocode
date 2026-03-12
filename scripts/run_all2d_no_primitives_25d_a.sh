#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiment.py -m \
    approach=agentic \
    approach.use_docker=true \
    approach.max_budget_usd=25 \
    seed=24 \
    'primitives=[]' \
    environment=clutteredstorage2d_medium,stickbutton2d_medium,obstruction2d_medium,clutteredretrieval2d_medium \
    'hydra.sweep.dir=multirun/2026-03-10/no_primitives_25d_all2d_s24' \
    'hydra.sweep.subdir=s${seed}/${hydra:runtime.choices.environment}'
