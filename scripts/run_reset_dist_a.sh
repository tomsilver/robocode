#!/bin/bash
# StickButton2D reset-distribution experiment, wave A.
#
# Requires third-party/kindergarden on branch exp/sb-reset-a (the stick spawn
# range is what differs). Runs one synthesis job per primitive condition at a
# single seed; eval_seed is pinned so every policy, here and in the frozen-policy
# cross-evaluations, faces an identical 100-instance suite.
set -euo pipefail

# shellcheck source=scripts/experiment_env.sh
source "$(dirname "$0")/experiment_env.sh"

OUT_ROOT="outputs/reset_dist_a_2026-08-01"
SEED=42
EVAL_SEED=1000
BUDGET=20.0

branch=$(git -C third-party/kindergarden branch --show-current)
if [ "$branch" != "exp/sb-reset-a" ]; then
    echo "kindergarden is on '$branch', expected exp/sb-reset-a" >&2
    exit 1
fi
git -C third-party/kindergarden rev-parse HEAD > /dev/null

mkdir -p "$OUT_ROOT"
git -C third-party/kindergarden rev-parse HEAD > "$OUT_ROOT/kindergarden_sha.txt"

for cond in noprims lowlevel bilevel; do
    run_dir="$OUT_ROOT/$cond/s$SEED"
    mkdir -p "$run_dir"
    echo "launching $cond -> $run_dir"
    nohup .venv/bin/python experiments/run_experiment.py \
        --config-name "$cond" \
        approach=agentic \
        approach/backend=claude_sonnet \
        approach.container_backend=apptainer \
        approach.max_budget_usd="$BUDGET" \
        approach.blackbox=false \
        num_eval_tasks=100 \
        seed="$SEED" \
        eval_seed="$EVAL_SEED" \
        environment=stickbutton2d_generalized \
        hydra.run.dir="$run_dir" \
        > "$run_dir/launch.log" 2>&1 &
    echo "  pid $!"
    sleep 90  # stagger so each run's source snapshot is taken before the next
done

wait
