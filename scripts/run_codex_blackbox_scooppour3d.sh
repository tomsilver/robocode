#!/usr/bin/env bash
# Rerun the five archived seeds with strict Docker Internet blocking.
# Settings come from Results/Final/Blackbox/Codex GPT5.6 Sol/dynamic3d.
# Requires the installed Python environment, Codex authentication, Docker,
# and the robocode-strict-blackbox image (see docs/blackbox.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

experiment_id=scooppour3d_generalized__agentic__none__blackbox__strict__codex_gpt56sol__timeout_60s__335203f0
run_dir="outputs/codex_strict_docker_rerun/$experiment_id/$(date +%Y-%m-%d_%H-%M-%S)"

# Match the archived count sweep, which differs from the current config defaults.
for seed in 42 24 222; do
    python experiments/run_experiment.py \
        environment=scooppour3d_generalized \
        'environment.design_counts=[10,20]' \
        'environment.eval_counts=[10,20,30,40,50]' \
        approach=agentic \
        primitive_level=none \
        approach.blackbox=true \
        approach.blackbox_strict=true \
        approach/backend=codex_gpt56sol \
        approach.container_backend=docker \
        approach.max_budget_usd=20.0 \
        num_eval_tasks=100 \
        max_steps=1000 \
        eval_timeout=60 \
        experiment_id="$experiment_id" \
        replicate_seed="$seed" \
        eval_seed=792075 \
        hydra.run.dir="$run_dir/replicate_$seed"
done
