#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TMPDIR="$PWD/.robocode-sandbox-mounts"
mkdir -p "$TMPDIR"

# Three environments, run serially. This worker stops on the first failure.
uv run python experiments/run_experiment.py -m environment=constrainedcupboard3d_generalized approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id=constrainedcupboard3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__54c3d8f7 replicate_seed=42,24,424,444,222 eval_seed=792075 'hydra.sweep.dir=multirun/constrainedcupboard3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__54c3d8f7/${now:%Y-%m-%d_%H-%M-%S}' 'hydra.sweep.subdir=replicate_${replicate_seed}'
uv run python experiments/run_experiment.py -m environment=sortclutteredblocks3d_generalized approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id=sortclutteredblocks3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__63fec73d replicate_seed=42,24,424,444,222 eval_seed=792075 'hydra.sweep.dir=multirun/sortclutteredblocks3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__63fec73d/${now:%Y-%m-%d_%H-%M-%S}' 'hydra.sweep.subdir=replicate_${replicate_seed}'
uv run python experiments/run_experiment.py -m environment=sweepintodrawer3d approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id=sweepintodrawer3d__llm_genplan__none__cli_opus5__timeout_60s__8adf7e50 replicate_seed=42,24,424,444,222 eval_seed=792075 'hydra.sweep.dir=multirun/sweepintodrawer3d__llm_genplan__none__cli_opus5__timeout_60s__8adf7e50/${now:%Y-%m-%d_%H-%M-%S}' 'hydra.sweep.subdir=replicate_${replicate_seed}'
