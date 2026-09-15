#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TMPDIR="$PWD/.robocode-sandbox-mounts"
mkdir -p "$TMPDIR"

# PR2Packed and PR2Blocked completed all five seeds. Rovers completed every seed
# except 222; keep this recovery script narrowly scoped so rerunning it cannot
# duplicate completed experiments.
uv run python experiments/run_experiment.py -m environment=rovers_generalized approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id=rovers_generalized__llm_genplan__none__cli_opus5__timeout_60s__37459045 replicate_seed=222 eval_seed=792075 'hydra.sweep.dir=multirun/rovers_generalized__llm_genplan__none__cli_opus5__timeout_60s__37459045/${now:%Y-%m-%d_%H-%M-%S}' 'hydra.sweep.subdir=replicate_${replicate_seed}'
