#!/bin/bash

# Submit one algorithm under the paper-aligned hidden 60 g meal, with matched
# no-Shield and predictive-Shield arms. The author-v1 meal-hyper check remains
# disabled in both the shared evaluator and the recorded predictive metadata.
set -euo pipefail

ALGORITHM="${1:-CPO}"
EPISODES="${2:-1}"
REPLICATE_ID="${3:-0}"
LOGIT_PENALTY="${4:-10}"
RESULT_COLLECTION="${5:-author_v1_unannounced_meal_evaluation}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PAIR_SCRIPT="${PROJECT_DIR}/scripts/submit_author_v1_stress_pair.sh"

test -f "${PAIR_SCRIPT}"

echo "Unannounced-meal protocol: 60 g daily at 15:00, post-action/pre-step"
echo "Predictive Shield meal-hyper check: false"

exec bash "${PAIR_SCRIPT}" \
    "${ALGORITHM}" \
    unannounced_meal_60g \
    "${EPISODES}" \
    "${REPLICATE_ID}" \
    "${LOGIT_PENALTY}" \
    "${RESULT_COLLECTION}"
