#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiment.py -m \
    approach=agentic \
    approach.use_docker=true \
    approach.max_budget_usd=25 \
    seed=24 \
    'primitives=[]' \
    environment=clutteredretrieval2d_hard,clutteredstorage2d_medium,clutteredstorage2d_hard,stickbutton2d_medium,stickbutton2d_hard \
    'hydra.sweep.dir=multirun/2026-02-25/no_primitives_25d_all2d_s24' \
    'hydra.sweep.subdir=s${seed}/${hydra:runtime.choices.environment}'
