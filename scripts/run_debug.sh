#!/bin/bash
# Run the launch.json "Record Approach History" config from the command line
set -e

python experiments/run_experiment.py \
    approach=agentic_cdl \
    approach.load_dir=outputs/cdl_no_mp_obstruction2d_hard_04-07/s24 \
    environment=obstruction2d_hard \
    seed=24 \
    "primitives=[]" \
    num_eval_tasks=100 \
    render_videos=true \
    hydra.run.dir=outputs/cdl_no_mp_obstruction2d_hard_04-07/s24
