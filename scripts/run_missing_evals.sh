#!/bin/bash
# Re-run evaluation for all runs missing results.json using approach.load_dir
set -e

run_eval() {
    local dir="$1"
    local env="$2"
    local seed="$3"
    local primitives="$4"

    echo ">>> Running: $dir (env=$env, seed=$seed, primitives=$primitives)"
    python experiments/run_experiment.py \
        approach=agentic_cdl \
        approach.use_docker=true \
        approach.max_budget_usd=20.0 \
        seed="$seed" \
        num_eval_tasks=100 \
        "primitives=$primitives" \
        'mcp_tools=[render_state,render_policy]' \
        environment="$env" \
        approach.load_dir="$dir" \
        "hydra.run.dir=$dir" 2>&1 | tail -3
    echo "<<< Done: $dir"
    echo ""
}

# With primitives (BiRRT)
# run_eval "outputs/cdl_mp_clutteredstorage2d_medium_04-01/s444" clutteredstorage2d_medium 444 "[BiRRT]"
# run_eval "outputs/cdl_mp_obstruction2d_hard_04-03/s24" obstruction2d_hard 24 "[BiRRT]"
# run_eval "outputs/cdl_mp_obstruction2d_hard_04-03/s42" obstruction2d_hard 42 "[BiRRT]"
# run_eval "outputs/cdl_mp_obstruction2d_hard_04-03/s444" obstruction2d_hard 444 "[BiRRT]"
# run_eval "outputs/cdl_mp_pushpullhook2d_04-01/s444" pushpullhook2d 444 "[BiRRT]"
# run_eval "outputs/cdl_mp_stickbutton2d_hard_04-03/s42" stickbutton2d_hard 42 "[BiRRT]"
# run_eval "outputs/cdl_mp_stickbutton2d_hard_04-03/s444" stickbutton2d_hard 444 "[BiRRT]"

# # Without primitives
# run_eval "outputs/cdl_no_mp_clutteredstorage2d_medium_2026-04-02/s24" clutteredstorage2d_medium 24 "[]"
# run_eval "outputs/cdl_no_mp_pushpullhook2d_2026-04-01/s24" pushpullhook2d 24 "[]"
# run_eval "outputs/cdl_no_mp_pushpullhook2d_2026-04-01/s42" pushpullhook2d 42 "[]"
# run_eval "outputs/cdl_no_mp_obstruction2d_hard_04-07/s24" obstruction2d_hard 24 "[]"
# run_eval "outputs/cdl_no_mp_obstruction2d_hard_04-07/s444" obstruction2d_hard 444 "[]"
# run_eval "outputs/cdl_no_mp_stickbutton2d_hard_04-07/s24" stickbutton2d_hard 24 "[]"

# With primitives (BiRRT) — obstruction2d_hard 04-08
# run_eval "outputs/cdl_mp_obstruction2d_hard_04-08/s24" obstruction2d_hard 24 "[BiRRT]"
# run_eval "outputs/cdl_mp_obstruction2d_hard_04-08/s42" obstruction2d_hard 42 "[BiRRT]"
# run_eval "outputs/cdl_mp_obstruction2d_hard_04-08/s444" obstruction2d_hard 444 "[BiRRT]"

# # Without primitives — obstruction2d_hard 04-07
# run_eval "outputs/cdl_no_mp_obstruction2d_hard_04-07/s24" obstruction2d_hard 24 "[]"
# run_eval "outputs/cdl_no_mp_obstruction2d_hard_04-07/s42" obstruction2d_hard 42 "[]"
# run_eval "outputs/cdl_no_mp_obstruction2d_hard_04-07/s444" obstruction2d_hard 444 "[]"

# With primitives (BiRRT) — stickbutton2d_hard 04-08
# run_eval "outputs/cdl_mp_stickbutton2d_hard_04-08/s24" stickbutton2d_hard 24 "[BiRRT]"
# run_eval "outputs/cdl_mp_stickbutton2d_hard_04-08/s42" stickbutton2d_hard 42 "[BiRRT]"
# run_eval "outputs/cdl_mp_stickbutton2d_hard_04-08/s444" stickbutton2d_hard 444 "[BiRRT]"

# # Without primitives — stickbutton2d_hard 04-07
# run_eval "outputs/cdl_no_mp_stickbutton2d_hard_04-07/s24" stickbutton2d_hard 24 "[]"
# run_eval "outputs/cdl_no_mp_stickbutton2d_hard_04-07/s42" stickbutton2d_hard 42 "[]"

# With primitives (BiRRT) — clutteredstorage2d_medium 04-09
# run_eval "outputs/cdl_mp_clutteredstorage2d_medium_04-09/s24" clutteredstorage2d_medium 24 "[BiRRT]"
# run_eval "outputs/cdl_mp_clutteredstorage2d_medium_04-09/s42" clutteredstorage2d_medium 42 "[BiRRT]"
# run_eval "outputs/cdl_mp_clutteredstorage2d_medium_04-09/s444" clutteredstorage2d_medium 444 "[BiRRT]"

# Without primitives — clutteredstorage2d_medium 04-09
run_eval "outputs/cdl_no_mp_clutteredstorage2d_medium_04-09/s24" clutteredstorage2d_medium 24 "[]"
# run_eval "outputs/cdl_no_mp_clutteredstorage2d_medium_04-09/s42" clutteredstorage2d_medium 42 "[]"
# run_eval "outputs/cdl_no_mp_clutteredstorage2d_medium_04-09/s444" clutteredstorage2d_medium 444 "[]"

echo "=== All evaluations complete ==="
