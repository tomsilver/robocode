#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiment.py -m \
    approach=agentic \
    approach.use_docker=true \
    seed=24,444 \
    'primitives=[]' \
    environment=motion2d_easy,obstruction2d_easy,clutteredretrieval2d_easy,clutteredstorage2d_easy,stickbutton2d_easy,pushpullhook2d \
    'hydra.sweep.dir=multirun/2026-02-23/no_primitives_5d_s24_444' \
    'hydra.sweep.subdir=s${seed}/${hydra:runtime.choices.environment}'
