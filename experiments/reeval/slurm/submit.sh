#!/bin/bash
# Plan the pending checkpoints and submit them as one SLURM job array.
#
# Usage: submit.sh <site.env> [worker plan options, e.g. --select final --match motion2d]
#
# The site file sets the cluster-specific values; see site/*.env.example. Every call
# plans from what is still missing in REEVAL_OUT, so rerunning it resumes a campaign.
set -euo pipefail

SITE_ENV="$(realpath "${1:?usage: submit.sh <site.env> [plan options]}")"
shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SITE_ENV}"
: "${REEVAL_CONTAINER:?}" "${REEVAL_SIF:?}" "${REEVAL_STORE:?}" "${REEVAL_OUT:?}"
: "${REEVAL_LOGS:?}" "${REEVAL_EVAL_SEED:?}" "${REEVAL_CPUS:?}" "${REEVAL_TIME:?}"
: "${REEVAL_NUM_SHARDS:?}"

mkdir -p "${REEVAL_OUT}/manifests" "${REEVAL_LOGS}"
MANIFEST="${REEVAL_OUT}/manifests/$(date +%Y%m%d-%H%M%S).json"

"${REEVAL_CONTAINER}" exec --cleanenv \
    --bind "${REEVAL_STORE}:${REEVAL_STORE}:ro,${REEVAL_OUT}" \
    --pwd /opt/robocode \
    "${REEVAL_SIF}" \
    python -m experiments.reeval.worker plan \
        --store "${REEVAL_STORE}" \
        --out "${REEVAL_OUT}" \
        --manifest "${MANIFEST}" \
        --num-shards "${REEVAL_NUM_SHARDS}" \
        --jobs "${REEVAL_CPUS}" \
        "$@"

args=(
    --array="0-$((REEVAL_NUM_SHARDS - 1))%${REEVAL_MAX_CONCURRENT:-${REEVAL_NUM_SHARDS}}"
    --cpus-per-task="${REEVAL_CPUS}"
    --time="${REEVAL_TIME}"
    --hint=nomultithread
    --output="${REEVAL_LOGS}/%x_%A_%a.out"
    --export="ALL,REEVAL_SITE_ENV=${SITE_ENV},REEVAL_MANIFEST=${MANIFEST}"
)
[ -n "${REEVAL_PARTITION:-}" ] && args+=(--partition="${REEVAL_PARTITION}")
[ -n "${REEVAL_ACCOUNT:-}" ] && args+=(--account="${REEVAL_ACCOUNT}")
[ -n "${REEVAL_QOS:-}" ] && args+=(--qos="${REEVAL_QOS}")
# shellcheck disable=SC2206
[ -n "${REEVAL_SBATCH_EXTRA:-}" ] && args+=(${REEVAL_SBATCH_EXTRA})

if [ -n "${REEVAL_DRY_RUN:-}" ]; then
    echo sbatch "${args[@]}" "${HERE}/reeval.sbatch"
else
    sbatch "${args[@]}" "${HERE}/reeval.sbatch"
fi
