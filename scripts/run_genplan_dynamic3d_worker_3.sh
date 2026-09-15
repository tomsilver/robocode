#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TMPDIR="$PWD/.robocode-sandbox-mounts"
mkdir -p "$TMPDIR"

# Six unfinished GenPlan rerun seeds, balanced by expected runtime. Each seed is
# a separate fail-fast invocation so all child processes exit between seeds.
run_seed() {
    local environment="$1"
    local experiment_id="$2"
    local seed="$3"
    uv run python experiments/run_experiment.py -m environment="$environment" approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id="$experiment_id" replicate_seed="$seed" eval_seed=792075 "hydra.sweep.dir=multirun/${experiment_id}/\${now:%Y-%m-%d_%H-%M-%S}" 'hydra.sweep.subdir=replicate_${replicate_seed}'
}

run_seed balancebeam3d balancebeam3d__llm_genplan__none__cli_opus5__timeout_60s__6bf31b55 222
run_seed tossing3d_generalized tossing3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__0c3e12c6 424
run_seed tossing3d_generalized tossing3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__0c3e12c6 444
run_seed tossing3d_generalized tossing3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__0c3e12c6 222
run_seed rearrange3d rearrange3d__llm_genplan__none__cli_opus5__timeout_60s__80185e71 222
run_seed sweepintodrawer3d sweepintodrawer3d__llm_genplan__none__cli_opus5__timeout_60s__8adf7e50 222
