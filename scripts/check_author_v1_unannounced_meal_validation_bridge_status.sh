#!/bin/bash

# One-shot status, completeness, and validation check for the all-algorithm
# unexpected-meal validation bridge. It makes one scheduler snapshot and never
# sleeps or polls.
set -euo pipefail

RESULT_COLLECTION="${1:-author_v1_unannounced_meal_validation_bridge_10seed}"
REPLICATE_ID="${2:-0}"
EPISODES="${3:-10}"
LOGIT_PENALTY="${4:-10}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
CHECK_SCRIPT="${PROJECT_DIR}/scripts/check_author_v1_unannounced_meal_status.sh"
ALGORITHMS=(CPO CUP FOCOPS OnCRPO PCPO PPOLag RCPO TRPOLag)
SCENARIOS=(nominal unannounced_meal_60g)

if [[ ! "${RESULT_COLLECTION}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RESULT_COLLECTION must be a safe directory name" >&2
    exit 2
fi
if [[ ! "${REPLICATE_ID}" =~ ^[0-9]+$ ]]; then
    echo "REPLICATE_ID must be nonnegative" >&2
    exit 2
fi
if [[ "${EPISODES}" != 10 ]]; then
    echo "The validation bridge expects exactly 10 independent episode seeds per cell" >&2
    exit 2
fi
if [[ ! "${LOGIT_PENALTY}" =~ ^[0-9]+$ ]]; then
    echo "LOGIT_PENALTY must be a nonnegative integer" >&2
    exit 2
fi
test -x "${CHECK_SCRIPT}"

job_id_file=$(mktemp)
trap 'rm -f "${job_id_file}"' EXIT

for scenario in "${SCENARIOS[@]}"; do
    record="${PROJECT_DIR}/${RESULT_COLLECTION}/${scenario}/replicate${REPLICATE_ID}/submitted_jobs.tsv"
    test -f "${record}"
    echo "Submission record: ${record}"
    column -t -s $'\t' "${record}"
    awk -F $'\t' 'NR > 1 {print $9}' "${record}" >> "${job_id_file}"
    echo
done

JOB_IDS=$(sort -u "${job_id_file}" | paste -sd, -)
if [[ -z "${JOB_IDS}" ]]; then
    echo "No validation-bridge job IDs found" >&2
    exit 1
fi

echo "Current queue snapshot (one query):"
squeue -j "${JOB_IDS}" -o "%.18i %.42j %.2t %.10M %R"

echo
echo "Accounting snapshot (one query):"
sacct -X -j "${JOB_IDS}" \
    --format=JobID,JobName%42,State,ExitCode,Elapsed

for scenario in "${SCENARIOS[@]}"; do
    for algorithm in "${ALGORITHMS[@]}"; do
        echo
        echo "===== ${scenario}: ${algorithm} ====="
        UNANNOUNCED_MEAL_CHECK_SCHEDULER=0 \
        UNANNOUNCED_MEAL_CHECK_SHOW_RECORD=0 \
            bash "${CHECK_SCRIPT}" \
            "${RESULT_COLLECTION}" "${REPLICATE_ID}" "${algorithm}" \
            "${EPISODES}" "${LOGIT_PENALTY}" "${scenario}"
    done
done

echo
echo "Validation-bridge one-shot check finished."
