#!/usr/bin/env bash
# Build the robocode-sandbox Docker image.
#
# Run from anywhere inside the repository:
#   bash docker/build.sh
#
# Rebuild when PyPI dependencies in pyproject.toml / uv.lock change.
# No rebuild needed for prpl-mono code changes (bind-mounted at runtime).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building robocode-sandbox from ${REPO_ROOT} ..."
docker build \
    --tag robocode-sandbox \
    --file "${REPO_ROOT}/docker/Dockerfile" \
    "${REPO_ROOT}"
echo "Done. Image tagged: robocode-sandbox"
