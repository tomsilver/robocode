#!/usr/bin/env bash
# Run cap-x's official launch.py against LIBERO-PRO using the robocode .venv.
# Assumes `source .venv/bin/activate` before invocation.
echo "sk-or-v1-your-key" > .openrouterkey
set -euo pipefail

uv run --active third-party/cap-x/capx/envs/launch.py \
    --config-path third-party/cap-x/env_configs/libero/franka_libero_cap_agent0.yaml \
    --total-trials 1
