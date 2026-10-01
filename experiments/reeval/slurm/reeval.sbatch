#!/bin/bash
# One array task: evaluate one manifest shard inside the robocode-eval image.
# Submitted by submit.sh, which passes REEVAL_SITE_ENV and REEVAL_MANIFEST.
#SBATCH --job-name=robocode-reeval
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail

# shellcheck source=/dev/null
source "${REEVAL_SITE_ENV}"

SCRATCH="${REEVAL_SCRATCH:-${TMPDIR:-/tmp}}/reeval-${SLURM_JOB_ID}"
mkdir -p "${SCRATCH}"
trap 'rm -rf "${SCRATCH}"' EXIT

# Passed through the container's environment, not its command line.
for prefix in APPTAINERENV_ SINGULARITYENV_; do
    export "${prefix}REEVAL_EVAL_SEED=${REEVAL_EVAL_SEED}"
    export "${prefix}SLURM_JOB_ID=${SLURM_JOB_ID}"
    export "${prefix}TMPDIR=${SCRATCH}"
done

echo "host $(hostname) cpus ${SLURM_CPUS_PER_TASK} shard ${SLURM_ARRAY_TASK_ID}"
"${REEVAL_CONTAINER}" exec --cleanenv \
    --bind "${REEVAL_STORE}:${REEVAL_STORE}:ro,${REEVAL_OUT},${SCRATCH}" \
    --pwd /opt/robocode \
    "${REEVAL_SIF}" \
    python -m experiments.reeval.worker run \
        --store "${REEVAL_STORE}" \
        --out "${REEVAL_OUT}" \
        --manifest "${REEVAL_MANIFEST}" \
        --shard "${SLURM_ARRAY_TASK_ID}" \
        --jobs "${SLURM_CPUS_PER_TASK}" \
        --scratch "${SCRATCH}"
