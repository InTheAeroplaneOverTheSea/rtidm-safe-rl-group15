#!/bin/bash

# This script only submits work to Slurm; it does not run evaluation locally.
set -euo pipefail

ALGORITHM="${1:?Usage: $0 ALGORITHM [episodes] [penalty] [result_collection]}"
EPISODES="${2:-1}"
LOGIT_PENALTY="${3:-10}"
RESULT_COLLECTION="${4:-author_v1_all_baselines_evaluation}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
JOB_SCRIPT="${PROJECT_DIR}/jobs/evaluate_author_v1_t1d_baseline.slurm"
RECORD="${PROJECT_DIR}/${RESULT_COLLECTION}/submitted_jobs.tsv"

case "${ALGORITHM}" in
    CPO|CUP|FOCOPS|OnCRPO|PCPO|PPOLag|RCPO|TRPOLag) ;;
    *) echo "Unsupported algorithm: ${ALGORITHM}" >&2; exit 2 ;;
esac
if [[ ! "${EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Episodes must be a positive integer, got ${EPISODES}" >&2
    exit 2
fi
if [[ ! "${LOGIT_PENALTY}" =~ ^[0-9]+$ ]]; then
    echo "Logit penalty must be a nonnegative integer, got ${LOGIT_PENALTY}" >&2
    exit 2
fi
if [[ ! "${RESULT_COLLECTION}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Result collection must be a single safe directory name" >&2
    exit 2
fi
mkdir -p "$(dirname "${RECORD}")"
expected_header=$'submitted_at\texperiment\treplicate_id\talgorithm\tcondition\tphase\tepisodes\tdevice_mode\tjob_id'
if [[ -f "${RECORD}" ]] && [[ "$(head -n 1 "${RECORD}")" != "${expected_header}" ]]; then
    echo "Refusing to append the new record format to legacy file ${RECORD}" >&2
    echo "Use a fresh RESULT_COLLECTION for dual-outcome runs." >&2
    exit 2
fi
if [[ -f "${RECORD}" ]] && awk -F '\t' -v algorithm="${ALGORITHM}" \
    '$4 == algorithm { found = 1 } END { exit(found ? 0 : 1) }' "${RECORD}"; then
    echo "Refusing duplicate submission: ${ALGORITHM} already has jobs in ${RECORD}" >&2
    exit 2
fi
if [[ ! -f "${RECORD}" ]]; then
    printf '%s\n' "${expected_header}" > "${RECORD}"
fi

algorithm_slug=$(printf '%s' "${ALGORITHM}" | tr '[:upper:]' '[:lower:]')
baseline_smoke_raw=$(sbatch --parsable \
    --job-name="v1-${algorithm_slug}-base-smoke" \
    --array=0-0 \
    "${JOB_SCRIPT}" "${ALGORITHM}" no_shield "${EPISODES}" "${LOGIT_PENALTY}" "${RESULT_COLLECTION}")
shield_smoke_raw=$(sbatch --parsable \
    --job-name="v1-${algorithm_slug}-shield-smoke" \
    --array=0-0 \
    "${JOB_SCRIPT}" "${ALGORITHM}" predictive "${EPISODES}" "${LOGIT_PENALTY}" "${RESULT_COLLECTION}")
baseline_smoke="${baseline_smoke_raw%%;*}"
shield_smoke="${shield_smoke_raw%%;*}"
baseline_full_raw=$(sbatch --parsable \
    --dependency="afterok:${baseline_smoke}" \
    --job-name="v1-${algorithm_slug}-base" \
    --array=1-80%18 \
    "${JOB_SCRIPT}" "${ALGORITHM}" no_shield "${EPISODES}" "${LOGIT_PENALTY}" "${RESULT_COLLECTION}")
shield_full_raw=$(sbatch --parsable \
    --dependency="afterok:${shield_smoke}" \
    --job-name="v1-${algorithm_slug}-shield" \
    --array=1-80%8 \
    "${JOB_SCRIPT}" "${ALGORITHM}" predictive "${EPISODES}" "${LOGIT_PENALTY}" "${RESULT_COLLECTION}")
baseline_full="${baseline_full_raw%%;*}"
shield_full="${shield_full_raw%%;*}"

submitted_at=$(date --iso-8601=seconds)
record_job() {
    local condition="$1"
    local phase="$2"
    local job_id="$3"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${submitted_at}" paper_baseline 0 "${ALGORITHM}" "${condition}" \
        "${phase}" "${EPISODES}" author_v1_cpu "${job_id}" >> "${RECORD}"
}
record_job no_shield smoke "${baseline_smoke}"
record_job predictive smoke "${shield_smoke}"
record_job no_shield full "${baseline_full}"
record_job predictive full "${shield_full}"

echo "${ALGORITHM} baseline smoke: ${baseline_smoke}"
echo "${ALGORITHM} Shield smoke:   ${shield_smoke}"
echo "${ALGORITHM} baseline full:  ${baseline_full} (starts after its smoke passes)"
echo "${ALGORITHM} Shield full:    ${shield_full} (starts after its smoke passes)"
echo "Recorded in: ${RECORD}"
