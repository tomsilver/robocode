#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building robocode-strict-blackbox from ${REPO_ROOT} ..."
docker build \
    --tag robocode-strict-blackbox \
    --file "${REPO_ROOT}/docker/Dockerfile.strict-blackbox" \
    --build-arg "USER_UID=$(id -u)" \
    --build-arg "USER_GID=$(id -g)" \
    "${REPO_ROOT}"
echo "Done. Image tagged: robocode-strict-blackbox"
