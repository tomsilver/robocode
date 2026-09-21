#!/usr/bin/env bash
# Bowen's Astra high / $20 assignments, Runs rows 72-81.
# Source: https://docs.google.com/spreadsheets/d/1v07OVUA9wOaclhfZ7oPeut4qpdcitLYljVR0vnveuWM/edit?gid=732132965
# Run alongside run_codex_blackbox_bowen_astra1.sh for two concurrent jobs.
# Each script runs its ten jobs sequentially and stops on failure.
# Requires the project Python environment, Codex authentication, Docker,
# and the robocode-strict-blackbox image (see docs/blackbox.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

runs=(
    "222 dynscooppour2d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__464dfcb9"
    "222 balancebeam3d__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__80c747e2"
    "222 pushpullhook2d__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__b7e2e339"
    "222 rovers_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__d2a8c165"
    "222 tossing3d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__f57080df"
    "222 clutteredstorage2d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__9b0745b7"
    "222 transport3d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__5f24c52e"
    "222 pr2packed_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__4e793a00"
    "222 dynpusht2d__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__c9fcf0a9"
    "222 motion2d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__154f05c7"
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
