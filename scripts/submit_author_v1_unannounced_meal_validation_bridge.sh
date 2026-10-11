#!/bin/bash

# Submit the validation bridge for all eight algorithms. The matched nominal
# and hidden-meal scenarios use the same corrected stress harness, episode
# count, replicate, policies, and Shield configuration.
set -euo pipefail

EPISODES="${1:-10}"
REPLICATE_ID="${2:-0}"
LOGIT_PENALTY="${3:-10}"
RESULT_COLLECTION="${4:-author_v1_unannounced_meal_validation_bridge_10seed}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PAIR_SCRIPT="${PROJECT_DIR}/scripts/submit_author_v1_stress_pair.sh"
ALGORITHMS=(CPO CUP FOCOPS OnCRPO PCPO PPOLag RCPO TRPOLag)
SCENARIOS=(nominal unannounced_meal_60g)

if [[ "${EPISODES}" != 10 ]]; then
    echo "The validation bridge must use exactly 10 independent episode seeds per cell" >&2
    exit 2
fi
if [[ ! "${REPLICATE_ID}" =~ ^[0-9]+$ ]]; then
    echo "REPLICATE_ID must be nonnegative" >&2
    exit 2
fi
if [[ ! "${LOGIT_PENALTY}" =~ ^[0-9]+$ ]]; then
    echo "LOGIT_PENALTY must be a nonnegative integer" >&2
    exit 2
fi
if [[ ! "${RESULT_COLLECTION}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RESULT_COLLECTION must be a safe directory name" >&2
    exit 2
fi
test -x "${PAIR_SCRIPT}"

for scenario in "${SCENARIOS[@]}"; do
    for algorithm in "${ALGORITHMS[@]}"; do
        bash "${PAIR_SCRIPT}" \
            "${algorithm}" "${scenario}" "${EPISODES}" \
            "${REPLICATE_ID}" "${LOGIT_PENALTY}" "${RESULT_COLLECTION}"
    done
done

echo "Validation bridge submitted for all eight algorithms."
echo "Result collection: ${RESULT_COLLECTION}"
echo "Episode seeds per cell: 22..31"
