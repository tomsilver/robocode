#!/bin/bash
# Run the launch.json "Record Approach History" config from the command line
set -e

python experiments/run_experiment.py \
    approach=agentic_cdl \
    approach.load_dir=analysis/cdl_mp_clutteredstorage2d_medium_04-01/s42 \
    environment=clutteredstorage2d_medium \
    seed=3 \
    "primitives=[BiRRT]" \
    num_eval_tasks=100 \
    render_videos=true \
    record_approach_history=false \
    hydra.run.dir=analysis/cdl_mp_clutteredstorage2d_medium_04-01/s42
