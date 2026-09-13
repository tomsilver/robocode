#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TMPDIR="$PWD/.robocode-sandbox-mounts"
mkdir -p "$TMPDIR"

# Three environments, run serially. This worker stops on the first failure.
uv run python experiments/run_experiment.py -m environment=dynamicshelf3d_generalized approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id=dynamicshelf3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__ed30005d replicate_seed=42,24,424,444,222 eval_seed=792075 'hydra.sweep.dir=multirun/dynamicshelf3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__ed30005d/${now:%Y-%m-%d_%H-%M-%S}' 'hydra.sweep.subdir=replicate_${replicate_seed}'
uv run python experiments/run_experiment.py -m environment=tossing3d_generalized approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id=tossing3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__0c3e12c6 replicate_seed=42,24,424,444,222 eval_seed=792075 'hydra.sweep.dir=multirun/tossing3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__0c3e12c6/${now:%Y-%m-%d_%H-%M-%S}' 'hydra.sweep.subdir=replicate_${replicate_seed}'
uv run python experiments/run_experiment.py -m environment=rearrange3d approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id=rearrange3d__llm_genplan__none__cli_opus5__timeout_60s__80185e71 replicate_seed=42,24,424,444,222 eval_seed=792075 'hydra.sweep.dir=multirun/rearrange3d__llm_genplan__none__cli_opus5__timeout_60s__80185e71/${now:%Y-%m-%d_%H-%M-%S}' 'hydra.sweep.subdir=replicate_${replicate_seed}'
