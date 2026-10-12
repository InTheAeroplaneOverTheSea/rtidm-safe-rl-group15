#!/bin/bash

# Submit one stress scenario for all eight author-v1 T1D algorithms.
set -euo pipefail

REQUESTED_STRESS_SCENARIO="${1:?Usage: $0 STRESS_SCENARIO [episodes] [replicate_id] [penalty] [result_collection]}"
case "${REQUESTED_STRESS_SCENARIO}" in
    sensor_sigma15) STRESS_SCENARIO="gaussian_sigma15" ;;
    *) STRESS_SCENARIO="${REQUESTED_STRESS_SCENARIO}" ;;
esac
EPISODES="${2:-1}"
REPLICATE_ID="${3:-0}"
LOGIT_PENALTY="${4:-10}"
RESULT_COLLECTION="${5:-author_v1_stress_evaluation}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PAIR_SCRIPT="${PROJECT_DIR}/scripts/submit_author_v1_stress_pair.sh"

case "${STRESS_SCENARIO}" in
    nominal|gaussian_sigma15|bias_positive|bias_negative|random_walk_drift|dropout_markov_v1|unannounced_meal_60g) ;;
    *) echo "Unsupported stress scenario: ${STRESS_SCENARIO}" >&2; exit 2 ;;
esac
test -x "${PAIR_SCRIPT}"

for algorithm in CPO CUP FOCOPS OnCRPO PCPO PPOLag RCPO TRPOLag; do
    bash "${PAIR_SCRIPT}" \
        "${algorithm}" \
        "${STRESS_SCENARIO}" \
        "${EPISODES}" \
        "${REPLICATE_ID}" \
        "${LOGIT_PENALTY}" \
        "${RESULT_COLLECTION}"
done
