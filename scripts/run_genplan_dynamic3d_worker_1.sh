#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TMPDIR="$PWD/.robocode-sandbox-mounts"
mkdir -p "$TMPDIR"

# Three environments, run serially. This worker stops on the first failure.
uv run python experiments/run_experiment.py -m environment=dynamo3d_generalized approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id=dynamo3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__b64ab8a9 replicate_seed=42,24,424,444,222 eval_seed=792075 'hydra.sweep.dir=multirun/dynamo3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__b64ab8a9/${now:%Y-%m-%d_%H-%M-%S}' 'hydra.sweep.subdir=replicate_${replicate_seed}'
uv run python experiments/run_experiment.py -m environment=scooppour3d_generalized approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id=scooppour3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__34f2bc34 replicate_seed=42,24,424,444,222 eval_seed=792075 'hydra.sweep.dir=multirun/scooppour3d_generalized__llm_genplan__none__cli_opus5__timeout_60s__34f2bc34/${now:%Y-%m-%d_%H-%M-%S}' 'hydra.sweep.subdir=replicate_${replicate_seed}'
uv run python experiments/run_experiment.py -m environment=balancebeam3d approach=llm_genplan primitive_level=none approach/completion=cli_opus5 eval_timeout=60 approach.max_budget_usd=20.0 approach.max_debug_attempts=null num_eval_tasks=100 experiment_id=balancebeam3d__llm_genplan__none__cli_opus5__timeout_60s__6bf31b55 replicate_seed=42,24,424,444,222 eval_seed=792075 'hydra.sweep.dir=multirun/balancebeam3d__llm_genplan__none__cli_opus5__timeout_60s__6bf31b55/${now:%Y-%m-%d_%H-%M-%S}' 'hydra.sweep.subdir=replicate_${replicate_seed}'
