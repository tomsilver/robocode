#!/usr/bin/env bash
set -euo pipefail

# Fixed evaluation suite from the reference constrainedcupboard3d campaign.
EVAL_SEED=792075
replicate_seeds=42,24,424,444,222
settings=(
    approach=agentic
    approach/backend=codex_gpt56sol
    approach.backend.reasoning_effort=medium
    approach.container_backend=apptainer
    approach.blackbox=true
    approach.blackbox_strict=true
    approach.max_budget_usd=20.0
    environment=tossing3d_generalized
    primitive_level=none
    num_eval_tasks=100
    eval_timeout=60
)
experiment_id=$(python - "$EVAL_SEED" "$replicate_seeds" "${settings[@]}" <<'PY'
import sys

import yaml

from experiments.tracker.constraints import ExperimentConfig
from experiments.tracker.generate import experiment_id

values = dict(arg.split("=", 1) for arg in sys.argv[3:])
config = ExperimentConfig(
    "tossing3d",
    {key: yaml.safe_load(value) for key, value in values.items()},
    tuple(map(int, sys.argv[2].split(","))),
)
print(experiment_id(config, int(sys.argv[1])))
PY
)

for seed in ${replicate_seeds//,/ }; do
    python experiments/run_experiment.py -m \
        "${settings[@]}" \
        "experiment_id=$experiment_id" \
        "replicate_seed=$seed" \
        "eval_seed=$EVAL_SEED" \
        'hydra.sweep.dir=Results/${experiment_id}/${now:%Y-%m-%d_%H-%M-%S}' \
        'hydra.sweep.subdir=replicate_${replicate_seed}'
done
