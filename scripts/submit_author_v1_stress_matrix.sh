#!/bin/bash

# Submit one stress scenario for all eight author-v1 T1D algorithms.
set -euo pipefail

STRESS_SCENARIO="${1:?Usage: $0 {nominal|sensor_sigma15|unannounced_meal_60g} [episodes] [replicate_id] [penalty] [result_collection]}"
EPISODES="${2:-1}"
REPLICATE_ID="${3:-0}"
LOGIT_PENALTY="${4:-10}"
RESULT_COLLECTION="${5:-author_v1_stress_evaluation}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PAIR_SCRIPT="${PROJECT_DIR}/scripts/submit_author_v1_stress_pair.sh"

case "${STRESS_SCENARIO}" in
    nominal|sensor_sigma15|unannounced_meal_60g) ;;
    *) echo "Unsupported stress scenario: ${STRESS_SCENARIO}" >&2; exit 2 ;;
esac
test -f "${PAIR_SCRIPT}"

for algorithm in CPO CUP FOCOPS OnCRPO PCPO PPOLag RCPO TRPOLag; do
    bash "${PAIR_SCRIPT}" \
        "${algorithm}" \
        "${STRESS_SCENARIO}" \
        "${EPISODES}" \
        "${REPLICATE_ID}" \
        "${LOGIT_PENALTY}" \
        "${RESULT_COLLECTION}"
done
