#!/bin/bash

# Submit one algorithm as paired no-Shield / predictive-Shield stress arrays.
set -euo pipefail

ALGORITHM="${1:?Usage: $0 ALGORITHM STRESS_SCENARIO [episodes] [replicate_id] [penalty] [result_collection]}"
REQUESTED_STRESS_SCENARIO="${2:?Usage: $0 ALGORITHM STRESS_SCENARIO [episodes] [replicate_id] [penalty] [result_collection]}"
case "${REQUESTED_STRESS_SCENARIO}" in
    sensor_sigma15) STRESS_SCENARIO="gaussian_sigma15" ;;
    *) STRESS_SCENARIO="${REQUESTED_STRESS_SCENARIO}" ;;
esac
EPISODES="${3:-1}"
REPLICATE_ID="${4:-0}"
LOGIT_PENALTY="${5:-10}"
RESULT_COLLECTION="${6:-author_v1_stress_evaluation}"
AFTEROK_JOB_ID="${STRESS_AFTEROK_JOB_ID:-}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
JOB_SCRIPT="${PROJECT_DIR}/jobs/evaluate_author_v1_t1d_stress.slurm"
RECORD_DIR="${PROJECT_DIR}/${RESULT_COLLECTION}/${STRESS_SCENARIO}/replicate${REPLICATE_ID}"
RECORD="${RECORD_DIR}/submitted_jobs.tsv"

case "${ALGORITHM}" in
    CPO|CUP|FOCOPS|OnCRPO|PCPO|PPOLag|RCPO|TRPOLag) ;;
    *) echo "Unsupported algorithm: ${ALGORITHM}" >&2; exit 2 ;;
esac
case "${STRESS_SCENARIO}" in
    nominal|gaussian_sigma15|bias_positive|bias_negative|random_walk_drift|dropout_markov_v1|unannounced_meal_60g) ;;
    *) echo "Unsupported stress scenario: ${STRESS_SCENARIO}" >&2; exit 2 ;;
esac
if [[ ! "${EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Episodes must be a positive integer, got ${EPISODES}" >&2
    exit 2
fi
if [[ ! "${REPLICATE_ID}" =~ ^[0-9]+$ ]]; then
    echo "Replicate id must be nonnegative, got ${REPLICATE_ID}" >&2
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
if [[ -n "${AFTEROK_JOB_ID}" ]] && [[ ! "${AFTEROK_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "STRESS_AFTEROK_JOB_ID must be one Slurm job id" >&2
    exit 2
fi
test -x "${JOB_SCRIPT}"
command -v sbatch >/dev/null

DEPENDENCY_ARGS=()
if [[ -n "${AFTEROK_JOB_ID}" ]]; then
    DEPENDENCY_ARGS+=(--dependency="afterok:${AFTEROK_JOB_ID}")
fi

mkdir -p "${RECORD_DIR}"
if [[ -f "${RECORD}" ]] && awk -F '\t' -v algorithm="${ALGORITHM}" \
    '$4 == algorithm { found = 1 } END { exit(found ? 0 : 1) }' "${RECORD}"; then
    echo "Refusing duplicate submission: ${ALGORITHM} already has jobs in ${RECORD}" >&2
    echo "Inspect the recorded jobs and retry only failed array cells manually." >&2
    exit 2
fi
if [[ ! -f "${RECORD}" ]]; then
    printf 'submitted_at\tstress_scenario\treplicate_id\talgorithm\tcondition\tphase\tepisodes\tdevice_mode\tjob_id\n' > "${RECORD}"
fi

submitted_at=$(date --iso-8601=seconds)
record_job() {
    local condition="$1"
    local phase="$2"
    local job_id="$3"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${submitted_at}" "${STRESS_SCENARIO}" "${REPLICATE_ID}" \
        "${ALGORITHM}" "${condition}" "${phase}" "${EPISODES}" \
        author_v1_cpu "${job_id}" >> "${RECORD}"
}

algorithm_slug=$(printf '%s' "${ALGORITHM}" | tr '[:upper:]' '[:lower:]')
case "${STRESS_SCENARIO}" in
    nominal) scenario_slug="nom" ;;
    gaussian_sigma15) scenario_slug="gauss15" ;;
    bias_positive) scenario_slug="biasp10" ;;
    bias_negative) scenario_slug="biasm10" ;;
    random_walk_drift) scenario_slug="drift003" ;;
    dropout_markov_v1) scenario_slug="gapmarkov" ;;
    unannounced_meal_60g) scenario_slug="hidden60" ;;
esac

common_args=(
    "${STRESS_SCENARIO}"
    "${EPISODES}"
    "${REPLICATE_ID}"
    "${LOGIT_PENALTY}"
    "${RESULT_COLLECTION}"
)

baseline_smoke_raw=$(sbatch --parsable \
    "${DEPENDENCY_ARGS[@]}" \
    --job-name="${scenario_slug}-${algorithm_slug}-base-smoke" \
    --array=0-0 \
    "${JOB_SCRIPT}" "${ALGORITHM}" no_shield "${common_args[@]}")
baseline_smoke="${baseline_smoke_raw%%;*}"
record_job no_shield smoke "${baseline_smoke}"

shield_smoke_raw=$(sbatch --parsable \
    "${DEPENDENCY_ARGS[@]}" \
    --job-name="${scenario_slug}-${algorithm_slug}-shield-smoke" \
    --array=0-0 \
    "${JOB_SCRIPT}" "${ALGORITHM}" predictive "${common_args[@]}")
shield_smoke="${shield_smoke_raw%%;*}"
record_job predictive smoke "${shield_smoke}"

baseline_full_raw=$(sbatch --parsable \
    --dependency="afterok:${baseline_smoke}" \
    --job-name="${scenario_slug}-${algorithm_slug}-base" \
    --array=1-80%18 \
    "${JOB_SCRIPT}" "${ALGORITHM}" no_shield "${common_args[@]}")
baseline_full="${baseline_full_raw%%;*}"
record_job no_shield full "${baseline_full}"

shield_full_raw=$(sbatch --parsable \
    --dependency="afterok:${shield_smoke}" \
    --job-name="${scenario_slug}-${algorithm_slug}-shield" \
    --array=1-80%8 \
    "${JOB_SCRIPT}" "${ALGORITHM}" predictive "${common_args[@]}")
shield_full="${shield_full_raw%%;*}"
record_job predictive full "${shield_full}"

echo "${ALGORITHM} ${STRESS_SCENARIO} baseline smoke: ${baseline_smoke}"
echo "${ALGORITHM} ${STRESS_SCENARIO} Shield smoke:   ${shield_smoke}"
echo "${ALGORITHM} ${STRESS_SCENARIO} baseline full:  ${baseline_full}"
echo "${ALGORITHM} ${STRESS_SCENARIO} Shield full:    ${shield_full}"
echo "Results: ${RECORD_DIR}"
