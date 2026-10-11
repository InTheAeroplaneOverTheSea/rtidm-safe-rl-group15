#!/bin/bash

# Submit one algorithm as paired no-Shield / predictive-Shield stress arrays.
set -euo pipefail

ALGORITHM="${1:?Usage: $0 ALGORITHM {nominal|sensor_sigma15|unannounced_meal_60g} [episodes] [replicate_id] [penalty] [result_collection]}"
STRESS_SCENARIO="${2:?Usage: $0 ALGORITHM {nominal|sensor_sigma15|unannounced_meal_60g} [episodes] [replicate_id] [penalty] [result_collection]}"
EPISODES="${3:-1}"
REPLICATE_ID="${4:-0}"
LOGIT_PENALTY="${5:-10}"
RESULT_COLLECTION="${6:-author_v1_stress_evaluation}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
JOB_SCRIPT="${PROJECT_DIR}/jobs/evaluate_author_v1_t1d_stress.slurm"
RECORD_DIR="${PROJECT_DIR}/${RESULT_COLLECTION}/${STRESS_SCENARIO}/replicate${REPLICATE_ID}"
RECORD="${RECORD_DIR}/submitted_jobs.tsv"

case "${ALGORITHM}" in
    CPO|CUP|FOCOPS|OnCRPO|PCPO|PPOLag|RCPO|TRPOLag) ;;
    *) echo "Unsupported algorithm: ${ALGORITHM}" >&2; exit 2 ;;
esac
case "${STRESS_SCENARIO}" in
    nominal|sensor_sigma15|unannounced_meal_60g) ;;
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
test -f "${JOB_SCRIPT}"

mkdir -p "${RECORD_DIR}"
if [[ ! -f "${RECORD}" ]]; then
    printf 'submitted_at\tstress_scenario\treplicate_id\talgorithm\tcondition\tphase\tepisodes\tdevice_mode\tjob_id\n' > "${RECORD}"
fi

algorithm_slug=$(printf '%s' "${ALGORITHM}" | tr '[:upper:]' '[:lower:]')
case "${STRESS_SCENARIO}" in
    nominal) scenario_slug="nom" ;;
    sensor_sigma15) scenario_slug="sensor15" ;;
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
    --job-name="${scenario_slug}-${algorithm_slug}-base-smoke" \
    --array=0-0 \
    "${JOB_SCRIPT}" "${ALGORITHM}" no_shield "${common_args[@]}")
shield_smoke_raw=$(sbatch --parsable \
    --job-name="${scenario_slug}-${algorithm_slug}-shield-smoke" \
    --array=0-0 \
    "${JOB_SCRIPT}" "${ALGORITHM}" predictive "${common_args[@]}")
baseline_smoke="${baseline_smoke_raw%%;*}"
shield_smoke="${shield_smoke_raw%%;*}"

baseline_full_raw=$(sbatch --parsable \
    --dependency="afterok:${baseline_smoke}" \
    --job-name="${scenario_slug}-${algorithm_slug}-base" \
    --array=1-80%18 \
    "${JOB_SCRIPT}" "${ALGORITHM}" no_shield "${common_args[@]}")
shield_full_raw=$(sbatch --parsable \
    --dependency="afterok:${shield_smoke}" \
    --job-name="${scenario_slug}-${algorithm_slug}-shield" \
    --array=1-80%8 \
    "${JOB_SCRIPT}" "${ALGORITHM}" predictive "${common_args[@]}")
baseline_full="${baseline_full_raw%%;*}"
shield_full="${shield_full_raw%%;*}"

submitted_at=$(date --iso-8601=seconds)
for record in \
    "no_shield smoke ${baseline_smoke}" \
    "predictive smoke ${shield_smoke}" \
    "no_shield full ${baseline_full}" \
    "predictive full ${shield_full}"; do
    read -r condition phase job_id <<< "${record}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${submitted_at}" "${STRESS_SCENARIO}" "${REPLICATE_ID}" \
        "${ALGORITHM}" "${condition}" "${phase}" "${EPISODES}" \
        author_v1_cpu "${job_id}" >> "${RECORD}"
done

echo "${ALGORITHM} ${STRESS_SCENARIO} baseline smoke: ${baseline_smoke}"
echo "${ALGORITHM} ${STRESS_SCENARIO} Shield smoke:   ${shield_smoke}"
echo "${ALGORITHM} ${STRESS_SCENARIO} baseline full:  ${baseline_full}"
echo "${ALGORITHM} ${STRESS_SCENARIO} Shield full:    ${shield_full}"
echo "Results: ${RECORD_DIR}"
