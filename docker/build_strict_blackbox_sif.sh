#!/usr/bin/env bash
# Build the robocode-strict-blackbox Apptainer/Singularity image (SIF).
#
# Same pipeline as docker/build_sif.sh, applied to docker/Dockerfile.strict-blackbox:
#   1. podman build  -> OCI image (rootless, no docker daemon needed)
#   2. podman save   -> docker-archive tarball
#   3. apptainer build -> SIF from the tarball
#
# Run from anywhere inside the repository:
#   bash docker/build_strict_blackbox_sif.sh
#
# Rebuild when the strict Dockerfile, its entrypoint, or the copied MCP proxy
# sources change: unlike the regular image, nothing is bind-mounted at runtime.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_PATH="${REPO_ROOT}/robocode-strict-blackbox.sif"
TMP_DIR="$(mktemp -d -t robocode-strict-sif-XXXXXX)"
TAR_PATH="${TMP_DIR}/robocode-strict-blackbox.tar"

cleanup() {
    rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

echo "[1/3] Building OCI image with podman from ${REPO_ROOT} ..."
podman build \
    --tag robocode-strict-blackbox \
    --file "${REPO_ROOT}/docker/Dockerfile.strict-blackbox" \
    --build-arg "USER_UID=$(id -u)" \
    --build-arg "USER_GID=$(id -g)" \
    "${REPO_ROOT}"

echo "[2/3] Saving OCI image to ${TAR_PATH} ..."
podman save robocode-strict-blackbox -o "${TAR_PATH}"

echo "[3/3] Converting to SIF at ${SIF_PATH} ..."
apptainer build --force "${SIF_PATH}" "docker-archive://${TAR_PATH}"

echo "Done. SIF written to: ${SIF_PATH}"
