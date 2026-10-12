#!/bin/bash

# Submit Person A/C's original-paper baseline and matched unexpected-meal
# bridge.  Every cell runs literal evaluation seeds 22..31 and records both
# true BG and the exposed CGM trace.  This script submits only; it never polls.
set -euo pipefail

PAPER_COLLECTION="${1:-author_v1_person_ac_paper_baseline_dual_10seed}"
STRESS_COLLECTION="${2:-author_v1_person_ac_unannounced_meal_dual_bridge_10seed}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BASELINE_SUBMITTER="${PROJECT_DIR}/scripts/submit_author_v1_baseline_pair.sh"
STRESS_SUBMITTER="${PROJECT_DIR}/scripts/submit_author_v1_stress_pair.sh"
EPISODES=10
REPLICATE_ID=0
LOGIT_PENALTY=10
ALGORITHMS=(CPO CUP PCPO PPOLag)
SCENARIOS=(nominal unannounced_meal_60g)

for collection in "${PAPER_COLLECTION}" "${STRESS_COLLECTION}"; do
    if [[ ! "${collection}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Collection names must be safe single directory names" >&2
        exit 2
    fi
done
if [[ "${PAPER_COLLECTION}" == "${STRESS_COLLECTION}" ]]; then
    echo "Paper and stress collections must be different" >&2
    exit 2
fi
test -x "${BASELINE_SUBMITTER}"
test -x "${STRESS_SUBMITTER}"
command -v sbatch >/dev/null

paper_record="${PROJECT_DIR}/${PAPER_COLLECTION}/submitted_jobs.tsv"
if [[ -e "${paper_record}" ]]; then
    echo "Refusing to reuse paper-baseline submission record: ${paper_record}" >&2
    exit 2
fi
for scenario in "${SCENARIOS[@]}"; do
    stress_record="${PROJECT_DIR}/${STRESS_COLLECTION}/${scenario}/replicate${REPLICATE_ID}/submitted_jobs.tsv"
    if [[ -e "${stress_record}" ]]; then
        echo "Refusing to reuse stress submission record: ${stress_record}" >&2
        exit 2
    fi
done

# Ten episodes are normally well below two hours, while this leaves headroom
# for slower predictive-Shield cells. Callers may override SBATCH_TIMELIMIT.
export SBATCH_TIMELIMIT="${SBATCH_TIMELIMIT:-02:00:00}"

echo "Submitting original-paper evaluator reproduction (dual trace)."
for algorithm in "${ALGORITHMS[@]}"; do
    bash "${BASELINE_SUBMITTER}" \
        "${algorithm}" "${EPISODES}" "${LOGIT_PENALTY}" "${PAPER_COLLECTION}"
done

echo "Submitting matched stress-harness nominal and unexpected-meal scenarios."
for scenario in "${SCENARIOS[@]}"; do
    for algorithm in "${ALGORITHMS[@]}"; do
        bash "${STRESS_SUBMITTER}" \
            "${algorithm}" "${scenario}" "${EPISODES}" \
            "${REPLICATE_ID}" "${LOGIT_PENALTY}" "${STRESS_COLLECTION}"
    done
done

echo
echo "Person A/C dual-outcome 10-seed jobs submitted."
echo "Paper collection: ${PAPER_COLLECTION}"
echo "Stress collection: ${STRESS_COLLECTION}"
echo "Episode seeds: 22..31"
echo "No scheduler polling was started."
