#!/bin/bash
# Re-run evaluation for all runs missing results.json using approach.load_dir
set -e
: "${EVAL_SEED:?Set EVAL_SEED to the private evaluation-suite seed}"

run_eval() {
    local dir="$1"
    local env="$2"
    local replicate_seed="$3"
    local primitive_level="$4"

    echo ">>> Running: $dir (env=$env, replicate_seed=$replicate_seed, primitive_level=$primitive_level)"
    python experiments/run_experiment.py \
        approach=agentic_cdl \
        approach.container_backend=docker \
        approach.max_budget_usd=20.0 \
        replicate_seed="$replicate_seed" \
        eval_seed="$EVAL_SEED" \
        num_eval_tasks=100 \
        primitive_level="$primitive_level" \
        'mcp_tools=[render_state,render_policy]' \
        environment="$env" \
        approach.load_dir="$dir" \
        "hydra.run.dir=$dir" 2>&1 | tail -3
    echo "<<< Done: $dir"
    echo ""
}

# # Without primitives
# run_eval "outputs/cdl_no_mp_clutteredstorage2d_medium_2026-04-02/s24" clutteredstorage2d_medium 24 none
# run_eval "outputs/cdl_no_mp_pushpullhook2d_2026-04-01/s24" pushpullhook2d 24 none
# run_eval "outputs/cdl_no_mp_pushpullhook2d_2026-04-01/s42" pushpullhook2d 42 none
# run_eval "outputs/cdl_no_mp_obstruction2d_hard_04-07/s24" obstruction2d_hard 24 none
# run_eval "outputs/cdl_no_mp_obstruction2d_hard_04-07/s444" obstruction2d_hard 444 none
# run_eval "outputs/cdl_no_mp_stickbutton2d_hard_04-07/s24" stickbutton2d_hard 24 none

# # Without primitives — obstruction2d_hard 04-07
# run_eval "outputs/cdl_no_mp_obstruction2d_hard_04-07/s24" obstruction2d_hard 24 none
# run_eval "outputs/cdl_no_mp_obstruction2d_hard_04-07/s42" obstruction2d_hard 42 none
# run_eval "outputs/cdl_no_mp_obstruction2d_hard_04-07/s444" obstruction2d_hard 444 none

# # Without primitives — stickbutton2d_hard 04-07
# run_eval "outputs/cdl_no_mp_stickbutton2d_hard_04-07/s24" stickbutton2d_hard 24 none
# run_eval "outputs/cdl_no_mp_stickbutton2d_hard_04-07/s42" stickbutton2d_hard 42 none

# Without primitives — clutteredstorage2d_medium 04-09
run_eval "outputs/cdl_no_mp_clutteredstorage2d_medium_04-09/s24" clutteredstorage2d_medium 24 none
# run_eval "outputs/cdl_no_mp_clutteredstorage2d_medium_04-09/s42" clutteredstorage2d_medium 42 none
# run_eval "outputs/cdl_no_mp_clutteredstorage2d_medium_04-09/s444" clutteredstorage2d_medium 444 none

echo "=== All evaluations complete ==="
