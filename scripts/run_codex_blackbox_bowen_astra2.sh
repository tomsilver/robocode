#!/usr/bin/env bash
# Bowen's Astra high / $20 assignments, Runs rows 51-55.
# Source: https://docs.google.com/spreadsheets/d/1v07OVUA9wOaclhfZ7oPeut4qpdcitLYljVR0vnveuWM/edit?gid=732132965
# Run alongside run_codex_blackbox_bowen_astra1.sh for two concurrent jobs.
# Each script runs its five jobs sequentially and stops on failure.
# Requires the project Python environment, Codex authentication, Docker,
# and the robocode-strict-blackbox image (see docs/blackbox.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

experiment_ids=(
    packing3d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__ba39bb6b
    clutteredretrieval2d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__30a50255
    dynpushpullhook2d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__ae02c288
    obstruction3d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__26394f99
    dynobstruction2d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__6e7ebec4
)

for experiment_id in "${experiment_ids[@]}"; do
    environment="${experiment_id%%__*}"

    printf '\nStarting %s (replicate seed 222)\n' "$environment"
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
        replicate_seed=222 \
        eval_seed=792075 \
        "hydra.sweep.dir=multirun/$experiment_id/"'${now:%Y-%m-%d_%H-%M-%S}' \
        'hydra.sweep.subdir=replicate_${replicate_seed}'
done
