#!/usr/bin/env bash
set -euo pipefail
: "${EVAL_SEED:?Set EVAL_SEED to the private evaluation-suite seed}"

python experiments/run_experiment.py -m \
    approach=agentic \
    approach.container_backend=docker \
    replicate_seed=24,444 \
    eval_seed="$EVAL_SEED" \
    'primitives=[]' \
    environment=motion2d_easy,obstruction2d_easy,clutteredretrieval2d_easy,clutteredstorage2d_easy,stickbutton2d_easy,pushpullhook2d \
    'hydra.sweep.dir=multirun/2026-02-23/no_primitives_5d_s24_444' \
    'hydra.sweep.subdir=r${replicate_seed}/${hydra:runtime.choices.environment}'
