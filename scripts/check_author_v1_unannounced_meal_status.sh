#!/bin/bash

# One-shot status and completeness check for one author-v1 stress pair. This
# script never loops, sleeps, or polls. Run it again manually later.
set -euo pipefail

RESULT_COLLECTION="${1:-author_v1_unannounced_meal_evaluation}"
REPLICATE_ID="${2:-0}"
ALGORITHM="${3:-CPO}"
EPISODES="${4:-1}"
LOGIT_PENALTY="${5:-10}"
SCENARIO="${6:-unannounced_meal_60g}"
CHECK_SCHEDULER="${UNANNOUNCED_MEAL_CHECK_SCHEDULER:-1}"
SHOW_RECORD="${UNANNOUNCED_MEAL_CHECK_SHOW_RECORD:-1}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="${PROJECT_DIR}/${RESULT_COLLECTION}/${SCENARIO}/replicate${REPLICATE_ID}"
RECORD="${ROOT}/submitted_jobs.tsv"
PENALTY_TAG="${LOGIT_PENALTY%.*}"
BASE_ROOT="${ROOT}/no_shield/t1d/${ALGORITHM}"
SHIELD_ROOT="${ROOT}/predictive_penalty${PENALTY_TAG}/t1d/${ALGORITHM}"
PYTHON_BIN="${RTIDM_VENV:-${HOME}/venvs/rtidm}/bin/python"
REPORT_DIR="${PROJECT_DIR}/validation_reports"

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
if [[ ! "${CHECK_SCHEDULER}" =~ ^[01]$ ]]; then
    echo "UNANNOUNCED_MEAL_CHECK_SCHEDULER must be 0 or 1" >&2
    exit 2
fi
if [[ ! "${SHOW_RECORD}" =~ ^[01]$ ]]; then
    echo "UNANNOUNCED_MEAL_CHECK_SHOW_RECORD must be 0 or 1" >&2
    exit 2
fi
case "${SCENARIO}" in
    nominal) REPORT_SCENARIO_TAG=nominal ;;
    unannounced_meal_60g) REPORT_SCENARIO_TAG=hidden60 ;;
    *) echo "Unsupported comparison scenario: ${SCENARIO}" >&2; exit 2 ;;
esac
case "${ALGORITHM}" in
    CPO|CUP|FOCOPS|OnCRPO|PCPO|PPOLag|RCPO|TRPOLag) ;;
    *) echo "Unsupported algorithm: ${ALGORITHM}" >&2; exit 2 ;;
esac

test -f "${RECORD}"
test -x "${PYTHON_BIN}"

if (( SHOW_RECORD )); then
    echo "Submission rows for ${ALGORITHM}: ${RECORD}"
    {
        head -n 1 "${RECORD}"
        awk -F $'\t' -v algorithm="${ALGORITHM}" \
            'NR > 1 && $4 == algorithm' "${RECORD}"
    } | column -t -s $'\t'
fi

JOB_IDS=$(awk -F $'\t' -v algorithm="${ALGORITHM}" \
    'NR > 1 && $4 == algorithm {print $9}' "${RECORD}" \
    | sort -u | paste -sd, -)
if [[ -z "${JOB_IDS}" ]]; then
    echo "No job IDs found for ${ALGORITHM} in ${RECORD}" >&2
    exit 1
fi

if (( CHECK_SCHEDULER )); then
    echo
    echo "Current queue snapshot (one query):"
    squeue -j "${JOB_IDS}" -o "%.18i %.42j %.2t %.10M %R"

    echo
    echo "Accounting snapshot (one query):"
    sacct -X -j "${JOB_IDS}" \
        --format=JobID,JobName%42,State,ExitCode,Elapsed
fi

count_files() {
    local root="$1"
    local pattern="$2"
    if [[ ! -d "${root}" ]]; then
        printf '0'
        return
    fi
    find "${root}" -type f -name "${pattern}" | wc -l | tr -d ' '
}

base_summaries=$(count_files "${BASE_ROOT}" summary_results.csv)
base_details=$(count_files "${BASE_ROOT}" detailed_results.csv)
base_runs=$(count_files "${BASE_ROOT}" run_metadata.txt)
base_stress=$(count_files "${BASE_ROOT}" stress_episode_metadata.json)
shield_summaries=$(count_files "${SHIELD_ROOT}" summary_results.csv)
shield_details=$(count_files "${SHIELD_ROOT}" detailed_results.csv)
shield_runs=$(count_files "${SHIELD_ROOT}" run_metadata.txt)
shield_stress=$(count_files "${SHIELD_ROOT}" stress_episode_metadata.json)

echo
echo "Result completeness (expected 81 of each):"
printf '  no Shield: summaries=%s details=%s run_metadata=%s stress_metadata=%s\n' \
    "${base_summaries}" "${base_details}" "${base_runs}" "${base_stress}"
printf '  predictive: summaries=%s details=%s run_metadata=%s stress_metadata=%s\n' \
    "${shield_summaries}" "${shield_details}" "${shield_runs}" "${shield_stress}"

counts=(
    "${base_summaries}" "${base_details}" "${base_runs}" "${base_stress}"
    "${shield_summaries}" "${shield_details}" "${shield_runs}" "${shield_stress}"
)
for count in "${counts[@]}"; do
    if (( count != 81 )); then
        echo
        echo "Results are not complete yet. This was a one-shot check; no polling was started."
        exit 0
    fi
done

mkdir -p "${REPORT_DIR}"
echo
echo "All result files are present; running validators."

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/aggregate_author_v1_t1d.py" \
    "${BASE_ROOT}" \
    --expected-episodes "${EPISODES}" \
    --algorithm "${ALGORITHM}" \
    --condition no_shield \
    --stress-scenario "${SCENARIO}" \
    --replicate-id "${REPLICATE_ID}" \
    --expected-result-collection "${RESULT_COLLECTION}" \
    | tee "${REPORT_DIR}/${ALGORITHM}_${REPORT_SCENARIO_TAG}_r${REPLICATE_ID}_no_shield.txt"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/aggregate_author_v1_t1d.py" \
    "${SHIELD_ROOT}" \
    --expected-episodes "${EPISODES}" \
    --algorithm "${ALGORITHM}" \
    --condition predictive \
    --logit-penalty "${LOGIT_PENALTY}" \
    --stress-scenario "${SCENARIO}" \
    --replicate-id "${REPLICATE_ID}" \
    --expected-result-collection "${RESULT_COLLECTION}" \
    | tee "${REPORT_DIR}/${ALGORITHM}_${REPORT_SCENARIO_TAG}_r${REPLICATE_ID}_predictive.txt"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/validate_author_v1_stress_pairing.py" \
    "${BASE_ROOT}" \
    "${SHIELD_ROOT}" \
    --expected-episodes "${EPISODES}" \
    --replicate-id "${REPLICATE_ID}" \
    --logit-penalty "${LOGIT_PENALTY}" \
    | tee "${REPORT_DIR}/${ALGORITHM}_${REPORT_SCENARIO_TAG}_r${REPLICATE_ID}_pairing.txt"

echo
echo "${SCENARIO} pair validation: PASS"
