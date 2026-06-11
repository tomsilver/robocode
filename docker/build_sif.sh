#!/usr/bin/env bash
# Build the robocode-sandbox Apptainer/Singularity image (SIF).
#
# Uses the EXISTING docker/Dockerfile (no separate definition file).
# Pipeline:
#   1. podman build  -> OCI image (rootless, no docker daemon needed)
#   2. podman save   -> docker-archive tarball
#   3. apptainer build -> SIF from the tarball
#
# Run from anywhere inside the repository:
#   bash docker/build_sif.sh
#
# Rebuild when PyPI dependencies in pyproject.toml / uv.lock change.
# No rebuild needed for src/ or third-party/kindergarden/ code changes
# (bind-mounted at runtime).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_PATH="${REPO_ROOT}/robocode-sandbox.sif"
TMP_DIR="$(mktemp -d -t robocode-sif-XXXXXX)"
TAR_PATH="${TMP_DIR}/robocode-sandbox.tar"

cleanup() {
    rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

echo "[1/3] Building OCI image with podman from ${REPO_ROOT} ..."
podman build \
    --tag robocode-sandbox \
    --file "${REPO_ROOT}/docker/Dockerfile" \
    --build-arg "USER_UID=$(id -u)" \
    --build-arg "USER_GID=$(id -g)" \
    "${REPO_ROOT}"

echo "[2/3] Saving OCI image to ${TAR_PATH} ..."
podman save robocode-sandbox -o "${TAR_PATH}"

echo "[3/3] Converting to SIF at ${SIF_PATH} ..."
apptainer build --force "${SIF_PATH}" "docker-archive://${TAR_PATH}"

echo "Done. SIF written to: ${SIF_PATH}"
