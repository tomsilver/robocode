#!/usr/bin/env bash
# Test generated pull-skill approaches against the failed + successful states.
# Usage: bash scripts/test_skill_pushpullhook2d_pull.sh [seed1 seed2 ...]
# Defaults to seeds 42 24 424 if none specified.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SWEEP_DIR="$REPO_ROOT/multirun/2026-03-16/skill_agentic_pushpullhook2d_pull"
FAILED_STATES="$REPO_ROOT/init_states/pushpullhook2d/pull"
SUCCESS_STATES="$REPO_ROOT/init_states/pushpullhook2d/successful"

SEEDS=("${@:-42 24 424}")
if [ $# -eq 0 ]; then SEEDS=(42 24 424); fi

for seed in "${SEEDS[@]}"; do
    approach="$SWEEP_DIR/s${seed}/pushpullhook2d/sandbox/approach.py"
    if [ ! -f "$approach" ]; then
        echo "SKIP: no approach.py for seed $seed"
        continue
    fi
    echo "=========================================="
    echo "Testing seed $seed: $approach"
    echo "=========================================="
    python "$SCRIPT_DIR/test_approach_generated.py" "$approach" \
        --state_dirs "$FAILED_STATES" "$SUCCESS_STATES"
done
