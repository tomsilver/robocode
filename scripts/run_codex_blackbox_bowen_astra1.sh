#!/usr/bin/env bash
# Bowen's Astra high / $20 assignments, Runs rows 62-71.
# Source: https://docs.google.com/spreadsheets/d/1v07OVUA9wOaclhfZ7oPeut4qpdcitLYljVR0vnveuWM/edit?gid=732132965
# Run alongside run_codex_blackbox_bowen_astra2.sh for two concurrent jobs.
# Each script runs its ten jobs sequentially and stops on failure.
# Requires the project Python environment, Codex authentication, Docker,
# and the robocode-strict-blackbox image (see docs/blackbox.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

runs=(
    "24 clutteredstorage2d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__7ffb9866"
    "24 transport3d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__7bddd27e"
    "24 pr2packed_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__464ab83f"
    "24 dynpusht2d__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__6fad45c9"
    "24 motion2d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__54829653"
    "24 obstruction2d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__d0148e6c"
    "24 stickbutton2d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__2d58bcf6"
    "24 basemotion3d__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__83f0bdfc"
    "24 table3d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__75abd470"
    "222 dynamo3d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__177cbe54"
)

for run in "${runs[@]}"; do
    read -r replicate_seed experiment_id <<< "$run"
    environment="${experiment_id%%__*}"

    printf '\nStarting %s (replicate seed %s)\n' "$environment" "$replicate_seed"
    python experiments/run_experiment.py -m \
        environment="$environment" \
        approach=agentic \
        primitive_level=none \
        approach.blackbox=true \
        approach.blackbox_strict=true \
        approach/backend=codex_gpt6 \
        approach.backend.reasoning_effort=high \
        approach.container_backend=docker \
        approach.max_budget_usd=20.0 \
        eval_timeout=60 \
        max_steps=1000 \
        num_eval_tasks=100 \
        experiment_id="$experiment_id" \
        replicate_seed="$replicate_seed" \
        eval_seed=792075 \
        "hydra.sweep.dir=multirun/$experiment_id/"'${now:%Y-%m-%d_%H-%M-%S}' \
        'hydra.sweep.subdir=replicate_${replicate_seed}'
done
