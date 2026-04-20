#!/usr/bin/env bash
# Start a second OpenRouter proxy on :8122 for cap-x's Molmo visual-grounding
# client (capx/integrations/vision/molmo.py), reusing .openrouterkey.
# Run in a dedicated terminal alongside scripts/start_openrouter_proxy.sh.
#
# Note: cap-x's molmo.py has `model_name: str = "allenai/Molmo2-8B"` which is
# the local vLLM model tag. To route through OpenRouter, edit that default to
# an OpenRouter-hosted Molmo slug (e.g. "allenai/molmo-7b-d:free") — grep
# openrouter.ai/models for the current name.
#
# Env overrides:
#   PORT  listen port (default 8122)
set -euo pipefail

PORT="${PORT:-8122}"

if [[ ! -f .openrouterkey ]]; then
    echo "[start_molmo_proxy] ERROR: .openrouterkey not found at repo root" >&2
    exit 1
fi

echo "[start_molmo_proxy] listening on :${PORT} (Ctrl-C to stop)"
exec python third-party/cap-x/capx/serving/openrouter_server.py \
    --key-file .openrouterkey --port "$PORT"
