#!/usr/bin/env bash
# Run cap-x's official launch.py across all 10 tasks of libero_object_swap and
# libero_object_task using robocode's .venv.
#
# Prereqs (in a separate terminal):
#   source .venv/bin/activate && bash scripts/start_openrouter_proxy.sh
#
# Env overrides:
#   TRIALS      trials per task (default 1; paper baseline is 20-50)
#   LOG_DIR     per-task log dir  (default outputs/capx_reproduce_logs)
set -euo pipefail

# Headless MuJoCo / OpenGL. cap-x's launch.py doesn't import kinder (which
# pins these for us elsewhere), so set them here before any mujoco import.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

TRIALS="${TRIALS:-1}"
LOG_DIR="${LOG_DIR:-outputs/capx_reproduce_logs}"
mkdir -p "$LOG_DIR"

# ---- Confirm OpenRouter proxy is up ----
port_up() { (echo > "/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1; }

if ! port_up 8110; then
    echo "[reproduce] ERROR: OpenRouter proxy not reachable on :8110." >&2
    echo "[reproduce] Start it first in another terminal:" >&2
    echo "[reproduce]   source .venv/bin/activate && bash scripts/start_openrouter_proxy.sh" >&2
    exit 1
fi

# ---- Iterate the 20 tasks ----
SUITES=(libero_object_swap libero_object_task)
for suite in "${SUITES[@]}"; do
    for task_id in 6; do
        cfg="third-party/cap-x/env_configs/libero/franka_${suite}_${task_id}.yaml"
        log="$LOG_DIR/${suite}_${task_id}.log"
        echo "[reproduce] === ${suite} task_id=${task_id} (trials=${TRIALS}) ==="
        uv run --active third-party/cap-x/capx/envs/launch.py \
            --config-path "$cfg" \
            --total-trials "$TRIALS" \
            2>&1 | tee "$log"
    done
done

echo "[reproduce] done; per-task logs under $LOG_DIR"
