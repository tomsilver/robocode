#!/usr/bin/env bash
# Build the robocode-eval Apptainer image (SIF) from docker/Dockerfile.eval.
#
# Usage: bash docker/build_eval_sif.sh <mimiclabs_scenes dir> [output.sif]
#
# The scenes directory is kinder's dynamic3d/models/assets/mimiclabs_scenes after
# its first download (meshes/ and textures/ are not in git). The image records the
# robocode and kinder revisions it was built from; build from a clean checkout.
set -euo pipefail

SCENES="${1:?usage: build_eval_sif.sh <mimiclabs_scenes dir> [output.sif]}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_PATH="${2:-${REPO_ROOT}/robocode-eval.sif}"
TAG=robocode-eval

if [ ! -d "${SCENES}/meshes" ] || [ ! -d "${SCENES}/textures" ]; then
    echo "error: ${SCENES} lacks meshes/ or textures/" >&2
    exit 1
fi
if [ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ]; then
    echo "warning: building from a checkout with uncommitted changes" >&2
fi
git -C "${REPO_ROOT}" submodule update --init third-party/kindergarden
git -C "${REPO_ROOT}" submodule update --init --recursive third-party/ss-pybullet
REVISION="robocode $(git -C "${REPO_ROOT}" rev-parse HEAD) kinder $(git -C "${REPO_ROOT}/third-party/kindergarden" rev-parse HEAD)"

docker build \
    --file "${REPO_ROOT}/docker/Dockerfile.eval" \
    --build-context "scenes=${SCENES}" \
    --build-arg "ROBOCODE_REVISION=${REVISION}" \
    --tag "${TAG}" \
    "${REPO_ROOT}"

# Apptainer stages the image in a temp dir; keep it next to the output, off /tmp.
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$(dirname "${SIF_PATH}")/.apptainer-tmp}"
mkdir -p "${APPTAINER_TMPDIR}"
trap 'rm -rf "${APPTAINER_TMPDIR}"' EXIT
apptainer build --force "${SIF_PATH}" "docker-daemon://${TAG}:latest"
echo "Built ${SIF_PATH} (${REVISION})"
