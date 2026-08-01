#!/bin/bash
# Evaluate the existing standard-trained StickButton2D policies on the current
# reset distribution, without re-running synthesis (approach.load_dir loads a
# frozen approach.py, so no agent and no LLM cost).
#
# eval_seed is pinned to the same value the synthesis runs use, so every policy
# faces an identical 100-instance suite.
set -euo pipefail

# shellcheck source=scripts/experiment_env.sh
source "$(dirname "$0")/experiment_env.sh"

OUT_ROOT="${1:-outputs/reset_dist_a_2026-08-01/crosseval_standard_trained}"
EVAL_SEED=1000
JOBS=5

JULY_A="outputs/validation_fix_eval_count_leak_2026-07-23/full_20"
JULY_B="outputs/validation_fix_eval_count_leak_2026-07-24_new/seeds_24_424"

run_one() {
    local cond=$1 seed=$2 src=$3
    local run_dir="$OUT_ROOT/$cond/s$seed"
    mkdir -p "$run_dir"
    .venv/bin/python experiments/run_experiment.py \
        --config-name "$cond" \
        approach=agentic \
        approach/backend=claude_sonnet \
        approach.container_backend=apptainer \
        approach.load_dir="$src" \
        approach.blackbox=false \
        num_eval_tasks=100 \
        seed="$seed" \
        eval_seed="$EVAL_SEED" \
        environment=stickbutton2d_generalized \
        hydra.run.dir="$run_dir" \
        > "$run_dir/eval.log" 2>&1 \
        && echo "done $cond s$seed" \
        || echo "FAILED $cond s$seed"
}

for cond in noprims lowlevel bilevel; do
    for seed in 42 24 424; do
        if [ "$seed" = "42" ]; then
            src="$JULY_A/$cond/stickbutton2d_generalized/s$seed"
        else
            src="$JULY_B/$cond/stickbutton2d_generalized/s$seed"
        fi
        if [ ! -f "$src/sandbox/approach.py" ]; then
            echo "missing policy: $src" >&2
            exit 1
        fi
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
        run_one "$cond" "$seed" "$src" &
    done
done

wait
echo "all cross-evaluations finished"
