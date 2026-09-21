#!/usr/bin/env bash
# Bowen's Astra high / $20 assignments, Runs rows 46-50.
# Source: https://docs.google.com/spreadsheets/d/1v07OVUA9wOaclhfZ7oPeut4qpdcitLYljVR0vnveuWM/edit?gid=732132965
# Run alongside run_codex_blackbox_bowen_astra2.sh for two concurrent jobs.
# Each script runs its five jobs sequentially and stops on failure.
# Requires the project Python environment, Codex authentication, Docker,
# and the robocode-strict-blackbox image (see docs/blackbox.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

experiment_ids=(
    scooppour3d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__f180770d
    sortclutteredblocks3d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__3f0e3ad6
    pr2blocked_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__27d91bd0
    rearrange3d__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__4da55ad4
    dynamicshelf3d_generalized__agentic__none__blackbox__strict__codex_gpt6__timeout_60s__351a1100
)

for experiment_id in "${experiment_ids[@]}"; do
    environment="${experiment_id%%__*}"

    extra_args=()
    case "$environment" in
        scooppour3d_generalized)
            extra_args=('environment.design_counts=[10,20]' 'environment.eval_counts=[10,20,30,40,50]')
            ;;
        dynamicshelf3d_generalized)
            extra_args=('environment.eval_counts=[1,2,3,4,6,8]')
            ;;
    esac

    printf '\nStarting %s (replicate seed 222)\n' "$environment"
    python experiments/run_experiment.py -m \
        environment="$environment" \
        "${extra_args[@]}" \
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
