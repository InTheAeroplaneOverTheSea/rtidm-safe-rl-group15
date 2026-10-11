#!/bin/bash

# One-shot status, completeness, and validation check for the unexpected-meal
# algorithms assigned to Person A and Person C. The script takes one scheduler
# snapshot, then performs local file checks; it never sleeps or polls.
set -euo pipefail

RESULT_COLLECTION="${1:-author_v1_unannounced_meal_evaluation}"
REPLICATE_ID="${2:-0}"
EPISODES="${3:-1}"
LOGIT_PENALTY="${4:-10}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
SCENARIO=unannounced_meal_60g
RECORD="${PROJECT_DIR}/${RESULT_COLLECTION}/${SCENARIO}/replicate${REPLICATE_ID}/submitted_jobs.tsv"
PREREQUISITE_RECORD="${PROJECT_DIR}/${RESULT_COLLECTION}/${SCENARIO}/replicate${REPLICATE_ID}/prerequisite_jobs.tsv"
CHECK_SCRIPT="${PROJECT_DIR}/scripts/check_author_v1_unannounced_meal_status.sh"
ALGORITHMS=(CPO CUP PCPO PPOLag)

if [[ ! "${RESULT_COLLECTION}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RESULT_COLLECTION must be a safe directory name" >&2
    exit 2
fi
if [[ ! "${REPLICATE_ID}" =~ ^[0-9]+$ ]]; then
    echo "REPLICATE_ID must be nonnegative" >&2
    exit 2
fi
if [[ ! "${EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EPISODES must be positive" >&2
    exit 2
fi
if [[ ! "${LOGIT_PENALTY}" =~ ^[0-9]+$ ]]; then
    echo "LOGIT_PENALTY must be a nonnegative integer" >&2
    exit 2
fi

test -f "${RECORD}"
test -x "${CHECK_SCRIPT}"

selected_algorithm() {
    case "$1" in
        CPO|CUP|PCPO|PPOLag) return 0 ;;
        *) return 1 ;;
    esac
}

echo "Person A/C submission rows: ${RECORD}"
{
    head -n 1 "${RECORD}"
    while IFS=$'\t' read -r submitted_at stress_scenario replicate_id \
        algorithm condition phase episodes device_mode job_id; do
        if selected_algorithm "${algorithm}"; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${submitted_at}" "${stress_scenario}" "${replicate_id}" \
                "${algorithm}" "${condition}" "${phase}" "${episodes}" \
                "${device_mode}" "${job_id}"
        fi
    done < <(tail -n +2 "${RECORD}")
} | column -t -s $'\t'

if [[ -f "${PREREQUISITE_RECORD}" ]]; then
    echo
    echo "Prerequisite jobs: ${PREREQUISITE_RECORD}"
    column -t -s $'\t' "${PREREQUISITE_RECORD}"
fi

JOB_IDS=$(
    {
        while IFS=$'\t' read -r _ _ _ algorithm _ _ _ _ job_id; do
            if selected_algorithm "${algorithm}"; then
                printf '%s\n' "${job_id}"
            fi
        done < <(tail -n +2 "${RECORD}")
        if [[ -f "${PREREQUISITE_RECORD}" ]]; then
            awk -F $'\t' 'NR > 1 {print $NF}' "${PREREQUISITE_RECORD}"
        fi
    } | sort -u | paste -sd, -
)

if [[ -z "${JOB_IDS}" ]]; then
    echo "No Person A/C job IDs found in ${RECORD}" >&2
    exit 1
fi

echo
echo "Current queue snapshot (one query):"
squeue -j "${JOB_IDS}" -o "%.18i %.42j %.2t %.10M %R"

echo
echo "Accounting snapshot (one query):"
sacct -X -j "${JOB_IDS}" \
    --format=JobID,JobName%42,State,ExitCode,Elapsed

for algorithm in "${ALGORITHMS[@]}"; do
    echo
    echo "===== ${algorithm} completeness and validation ====="
    UNANNOUNCED_MEAL_CHECK_SCHEDULER=0 \
    UNANNOUNCED_MEAL_CHECK_SHOW_RECORD=0 \
        bash "${CHECK_SCRIPT}" \
        "${RESULT_COLLECTION}" "${REPLICATE_ID}" "${algorithm}" \
        "${EPISODES}" "${LOGIT_PENALTY}"
done

echo
echo "Person A/C unexpected-meal one-shot check finished."
