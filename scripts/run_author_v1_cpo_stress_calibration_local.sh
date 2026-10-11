#!/bin/bash

# Run the same one-cell CPO calibration as the SURF submitter, using the local
# virtual environment and bounded process parallelism. The worker delegates to
# the Slurm evaluator with harmless local shims for `module` and `srun` so the
# two execution paths cannot silently diverge.
set -euo pipefail

if [[ "${1:-}" == "--worker" ]]; then
    shift
    EPISODES="${1:?worker episodes}"
    REPLICATE_ID="${2:?worker replicate id}"
    LOGIT_PENALTY="${3:?worker penalty}"
    RESULT_COLLECTION="${4:?worker result collection}"
    SCENARIO="${5:?worker scenario}"
    CONDITION="${6:?worker condition}"

    PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    JOB_SCRIPT="${PROJECT_DIR}/jobs/evaluate_author_v1_t1d_stress.slurm"
    LOG_DIR="${PROJECT_DIR}/${RESULT_COLLECTION}/local_driver_logs"
    STATUS_FILE="${LOG_DIR}/status.tsv"
    mkdir -p "${LOG_DIR}"
    CELL_LOG="${LOG_DIR}/${SCENARIO}_${CONDITION}.log"
    WORK_TMP="$(mktemp -d "${TMPDIR:-/tmp}/rtidm-local-calibration.XXXXXX")"
    cleanup() {
        rm -rf -- "${WORK_TMP}"
    }
    trap cleanup EXIT

    module() { :; }
    srun() { "$@"; }
    export -f module srun
    export SLURM_SUBMIT_DIR="${PROJECT_DIR}"
    export SLURM_ARRAY_TASK_ID=0
    export SLURM_ARRAY_JOB_ID="local-${SCENARIO}-${CONDITION}"
    export RTIDM_VENV="${PROJECT_DIR}/.venv"
    export TMPDIR="${WORK_TMP}"
    export MPLCONFIGDIR="${LOG_DIR}/matplotlib_cache"
    mkdir -p "${MPLCONFIGDIR}"

    started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    status=PASS
    if ! bash "${JOB_SCRIPT}" CPO "${CONDITION}" "${SCENARIO}" \
        "${EPISODES}" "${REPLICATE_ID}" "${LOGIT_PENALTY}" \
        "${RESULT_COLLECTION}" > "${CELL_LOG}" 2>&1; then
        status=FAIL
    fi
    finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${started_at}" "${finished_at}" "${SCENARIO}" "${CONDITION}" \
        "${status}" >> "${STATUS_FILE}"
    echo "${SCENARIO} ${CONDITION}: ${status} (${CELL_LOG})"
    [[ "${status}" == PASS ]]
    exit
fi

EPISODES="${1:-10}"
REPLICATE_ID="${2:-0}"
LOGIT_PENALTY="${3:-10}"
RESULT_COLLECTION="${4:-author_v1_stress_calibration_local}"
MAX_PARALLEL="${5:-2}"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/${RESULT_COLLECTION}/local_driver_logs"
STATUS_FILE="${LOG_DIR}/status.tsv"
PLAN_FILE="${LOG_DIR}/plan.txt"

for value_name in EPISODES MAX_PARALLEL; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got ${value}" >&2
        exit 2
    fi
done
if [[ ! "${REPLICATE_ID}" =~ ^[0-9]+$ ]]; then
    echo "REPLICATE_ID must be nonnegative, got ${REPLICATE_ID}" >&2
    exit 2
fi
if [[ ! "${RESULT_COLLECTION}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RESULT_COLLECTION must be a safe directory name" >&2
    exit 2
fi

for required in \
    "${PROJECT_DIR}/.venv/bin/activate" \
    "${PROJECT_DIR}/author_v1_GlucoAlg_50d3134.tar.gz" \
    "${PROJECT_DIR}/author_v1_GlucoSim_6522d49.tar.gz" \
    "${PROJECT_DIR}/author_v1_all_baselines_evaluation/t1d_checkpoint_manifest.json" \
    "${PROJECT_DIR}/external/GlucoAlg/saved_models/t1d/child/CPO/seed0/config.json" \
    "${PROJECT_DIR}/external/GlucoAlg/saved_models/t1d/child/CPO/seed0/torch_save/epoch-2441.pt"; do
    test -f "${required}"
done

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

mkdir -p "${LOG_DIR}"
: > "${PLAN_FILE}"
printf 'started_at_utc\tfinished_at_utc\tstress_scenario\tcondition\tstatus\n' \
    > "${STATUS_FILE}"
for scenario in "${SCENARIOS[@]}"; do
    for condition in no_shield predictive; do
        printf '%s %s\n' "${scenario}" "${condition}" >> "${PLAN_FILE}"
    done
done

echo "Local CPO sensor calibration: 16 cells, ${EPISODES} episodes per cell"
echo "Parallel workers: ${MAX_PARALLEL}"
echo "Results: ${PROJECT_DIR}/${RESULT_COLLECTION}"

xargs -n 2 -P "${MAX_PARALLEL}" \
    bash "$0" --worker "${EPISODES}" "${REPLICATE_ID}" \
    "${LOGIT_PENALTY}" "${RESULT_COLLECTION}" < "${PLAN_FILE}"

echo "Local CPO calibration workers completed."
python "${PROJECT_DIR}/scripts/analyze_author_v1_cpo_stress_calibration.py" \
    "${PROJECT_DIR}/${RESULT_COLLECTION}"
