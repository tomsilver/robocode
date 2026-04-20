#!/usr/bin/env bash
# Start cap-x's OpenRouter proxy on :8110 in the foreground.
# Run this in a dedicated terminal before scripts/reproduce_capx_official.sh.
#
# Assumes `source .venv/bin/activate` and an OpenRouter API key at
# `.openrouterkey` in the repo root.
#
# Env overrides:
#   PORT  listen port (default 8110; must match cap-x configs)
set -euo pipefail

PORT="${PORT:-8110}"

if [[ ! -f .openrouterkey ]]; then
    echo "[start_openrouter_proxy] ERROR: .openrouterkey not found at repo root" >&2
    exit 1
fi

echo "[start_openrouter_proxy] listening on :${PORT} (Ctrl-C to stop)"
exec python third-party/cap-x/capx/serving/openrouter_server.py \
    --key-file .openrouterkey --port "$PORT"
