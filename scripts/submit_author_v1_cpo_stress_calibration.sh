#!/bin/bash

# Submit only the small CPO calibration gate: one patient/policy cell, both
# Shield arms, ten literal episode seeds (22..31 by default), and no full arrays.
set -euo pipefail

EPISODES="${1:-10}"
REPLICATE_ID="${2:-0}"
LOGIT_PENALTY="${3:-10}"
RESULT_COLLECTION="${4:-author_v1_stress_calibration}"
TASK_INDEX="${CALIBRATION_TASK_INDEX:-0}"  # child#002, CPO policy seed 0
AFTEROK_JOB_ID="${CALIBRATION_AFTEROK_JOB_ID:-}"
TIME_LIMIT="${CALIBRATION_TIME_LIMIT:-08:00:00}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
JOB_SCRIPT="${PROJECT_DIR}/jobs/evaluate_author_v1_t1d_stress.slurm"
RECORD_DIR="${PROJECT_DIR}/${RESULT_COLLECTION}"
RECORD="${RECORD_DIR}/submitted_jobs.tsv"

SCENARIOS=(
    nominal
    gaussian_sigma15
    bias_positive
    bias_negative
    drift_sigma0025_calibration
    random_walk_drift
    drift_sigma004_calibration
    dropout_markov_v1
)

if [[ ! "${EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Episodes must be a positive integer, got ${EPISODES}" >&2
    exit 2
fi
if [[ ! "${REPLICATE_ID}" =~ ^[0-9]+$ ]]; then
    echo "Replicate id must be nonnegative, got ${REPLICATE_ID}" >&2
    exit 2
fi
if [[ ! "${TASK_INDEX}" =~ ^[0-9]+$ ]] || (( TASK_INDEX > 80 )); then
    echo "CALIBRATION_TASK_INDEX must be between 0 and 80, got ${TASK_INDEX}" >&2
    exit 2
fi
if [[ ! "${RESULT_COLLECTION}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Result collection must be a safe directory name" >&2
    exit 2
fi
if [[ ! "${LOGIT_PENALTY}" =~ ^[0-9]+$ ]]; then
    echo "Logit penalty must be a nonnegative integer, got ${LOGIT_PENALTY}" >&2
    exit 2
fi
if [[ -n "${AFTEROK_JOB_ID}" ]] && [[ ! "${AFTEROK_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "CALIBRATION_AFTEROK_JOB_ID must be one Slurm job id" >&2
    exit 2
fi
if [[ ! "${TIME_LIMIT}" =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "CALIBRATION_TIME_LIMIT must use [days-]HH:MM:SS, got ${TIME_LIMIT}" >&2
    exit 2
fi
test -x "${JOB_SCRIPT}"
command -v sbatch >/dev/null

DEPENDENCY_ARGS=()
if [[ -n "${AFTEROK_JOB_ID}" ]]; then
    DEPENDENCY_ARGS+=("--dependency=afterok:${AFTEROK_JOB_ID}")
fi

mkdir -p "${RECORD_DIR}"
if [[ -e "${RECORD}" ]]; then
    echo "Refusing to append to existing calibration record: ${RECORD}" >&2
    echo "Use a new result collection or inspect/retry failed jobs manually." >&2
    exit 2
fi
printf 'submitted_at\tstress_scenario\treplicate_id\talgorithm\tcondition\tphase\tepisodes\ttask_index\tjob_id\n' > "${RECORD}"

for scenario in "${SCENARIOS[@]}"; do
    case "${scenario}" in
        nominal) slug="nom" ;;
        gaussian_sigma15) slug="gauss15" ;;
        bias_positive) slug="biasp10" ;;
        bias_negative) slug="biasm10" ;;
        drift_sigma0025_calibration) slug="drift0025" ;;
        random_walk_drift) slug="drift003" ;;
        drift_sigma004_calibration) slug="drift004" ;;
        dropout_markov_v1) slug="gapmarkov" ;;
    esac

    for condition in no_shield predictive; do
        job_raw=$(sbatch --parsable \
            "${DEPENDENCY_ARGS[@]}" \
            --time="${TIME_LIMIT}" \
            --job-name="cal-cpo-${slug}-${condition}" \
            --array="${TASK_INDEX}-${TASK_INDEX}" \
            "${JOB_SCRIPT}" CPO "${condition}" "${scenario}" \
            "${EPISODES}" "${REPLICATE_ID}" "${LOGIT_PENALTY}" \
            "${RESULT_COLLECTION}")
        job_id="${job_raw%%;*}"
        submitted_at=$(date --iso-8601=seconds)
        printf '%s\t%s\t%s\tCPO\t%s\tcalibration\t%s\t%s\t%s\n' \
            "${submitted_at}" "${scenario}" "${REPLICATE_ID}" \
            "${condition}" "${EPISODES}" "${TASK_INDEX}" "${job_id}" \
            >> "${RECORD}"
        echo "${scenario} ${condition}: ${job_id}"
    done
done

echo "Calibration jobs recorded in ${RECORD}"
echo "Calibration walltime per job: ${TIME_LIMIT}"
echo "This script submitted no full 81-cell arrays."
