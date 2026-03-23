#!/usr/bin/env bash
# Build the robocode-sandbox Docker image.
#
# Run from anywhere inside the repository:
#   bash docker/build.sh
#
# src/ and prpl-mono/ are bind-mounted at runtime, so rebuilds are only
# needed when system packages or Claude Code version change.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building robocode-sandbox from ${REPO_ROOT} ..."
docker build \
    --tag robocode-sandbox \
    --file "${REPO_ROOT}/docker/Dockerfile" \
    "${REPO_ROOT}"
echo "Done. Image tagged: robocode-sandbox"
