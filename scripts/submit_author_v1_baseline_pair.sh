#!/bin/bash

# This script only submits work to Slurm; it does not run evaluation locally.
set -euo pipefail

ALGORITHM="${1:?Usage: $0 ALGORITHM}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
JOB_SCRIPT="${PROJECT_DIR}/jobs/evaluate_author_v1_t1d_baseline.slurm"
RECORD="${PROJECT_DIR}/author_v1_all_baselines_evaluation/submitted_jobs.tsv"

case "${ALGORITHM}" in
    CPO|CUP|FOCOPS|OnCRPO|PCPO|PPOLag|RCPO|TRPOLag) ;;
    *) echo "Unsupported algorithm: ${ALGORITHM}" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "${RECORD}")"
if [[ ! -f "${RECORD}" ]]; then
    printf 'submitted_at\talgorithm\tcondition\tphase\tdevice_mode\tjob_id\n' > "${RECORD}"
fi

algorithm_slug=$(printf '%s' "${ALGORITHM}" | tr '[:upper:]' '[:lower:]')
baseline_smoke_raw=$(sbatch --parsable \
    --job-name="v1-${algorithm_slug}-base-smoke" \
    --array=0-0 \
    "${JOB_SCRIPT}" "${ALGORITHM}" no_shield 1 10)
shield_smoke_raw=$(sbatch --parsable \
    --job-name="v1-${algorithm_slug}-shield-smoke" \
    --array=0-0 \
    "${JOB_SCRIPT}" "${ALGORITHM}" predictive 1 10)
baseline_smoke="${baseline_smoke_raw%%;*}"
shield_smoke="${shield_smoke_raw%%;*}"
baseline_full_raw=$(sbatch --parsable \
    --dependency="afterok:${baseline_smoke}" \
    --job-name="v1-${algorithm_slug}-base" \
    --array=1-80%18 \
    "${JOB_SCRIPT}" "${ALGORITHM}" no_shield 1 10)
shield_full_raw=$(sbatch --parsable \
    --dependency="afterok:${shield_smoke}" \
    --job-name="v1-${algorithm_slug}-shield" \
    --array=1-80%8 \
    "${JOB_SCRIPT}" "${ALGORITHM}" predictive 1 10)
baseline_full="${baseline_full_raw%%;*}"
shield_full="${shield_full_raw%%;*}"

submitted_at=$(date --iso-8601=seconds)
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${submitted_at}" "${ALGORITHM}" no_shield smoke author_v1_cpu "${baseline_smoke}" >> "${RECORD}"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${submitted_at}" "${ALGORITHM}" predictive smoke author_v1_cpu "${shield_smoke}" >> "${RECORD}"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${submitted_at}" "${ALGORITHM}" no_shield full author_v1_cpu "${baseline_full}" >> "${RECORD}"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${submitted_at}" "${ALGORITHM}" predictive full author_v1_cpu "${shield_full}" >> "${RECORD}"

echo "${ALGORITHM} baseline smoke: ${baseline_smoke}"
echo "${ALGORITHM} Shield smoke:   ${shield_smoke}"
echo "${ALGORITHM} baseline full:  ${baseline_full} (starts after its smoke passes)"
echo "${ALGORITHM} Shield full:    ${shield_full} (starts after its smoke passes)"
echo "Recorded in: ${RECORD}"
