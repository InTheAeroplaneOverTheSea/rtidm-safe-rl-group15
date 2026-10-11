#!/bin/bash

# Run the smallest executable hidden-meal comparison locally: CPO,
# child#002, policy seed 0, with and without the predictive Shield.
set -euo pipefail

EPISODES="${1:-2}"
REPLICATE_ID="${2:-0}"
LOGIT_PENALTY="${3:-10}"
RESULT_COLLECTION="${4:-author_v1_unannounced_meal_local_smoke}"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER="${PROJECT_DIR}/scripts/run_author_v1_cpo_stress_calibration_local.sh"
ANALYZER="${PROJECT_DIR}/scripts/analyze_author_v1_cpo_stress_calibration.py"
PAIR_VALIDATOR="${PROJECT_DIR}/scripts/validate_author_v1_stress_pairing.py"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
RESULT_ROOT="${PROJECT_DIR}/${RESULT_COLLECTION}"
LOG_DIR="${RESULT_ROOT}/local_driver_logs"

if [[ ! "${EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EPISODES must be a positive integer, got ${EPISODES}" >&2
    exit 2
fi
if [[ ! "${REPLICATE_ID}" =~ ^[0-9]+$ ]]; then
    echo "REPLICATE_ID must be nonnegative, got ${REPLICATE_ID}" >&2
    exit 2
fi
if [[ ! "${RESULT_COLLECTION}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RESULT_COLLECTION must be a safe directory name" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" ]]; then
    echo "Result collection already exists: ${RESULT_ROOT}" >&2
    echo "Choose a new RESULT_COLLECTION to avoid mixing runs." >&2
    exit 2
fi

for required in "${WORKER}" "${ANALYZER}" "${PAIR_VALIDATOR}" "${PYTHON_BIN}"; do
    test -f "${required}"
done

mkdir -p "${LOG_DIR}"
printf 'started_at_utc\tfinished_at_utc\tstress_scenario\tcondition\tstatus\n' \
    > "${LOG_DIR}/status.tsv"

echo "Local CPO unannounced-meal pair"
echo "  cell: child#002 / policy seed 0"
echo "  episodes per arm: ${EPISODES}"
echo "  predictive meal-hyper check: false"
echo "  results: ${RESULT_ROOT}"

pids=()
for condition in no_shield predictive; do
    bash "${WORKER}" --worker \
        "${EPISODES}" \
        "${REPLICATE_ID}" \
        "${LOGIT_PENALTY}" \
        "${RESULT_COLLECTION}" \
        unannounced_meal_60g \
        "${condition}" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
if (( failed )); then
    echo "At least one local hidden-meal arm failed; inspect ${LOG_DIR}." >&2
    exit 1
fi

"${PYTHON_BIN}" "${ANALYZER}" \
    "${RESULT_ROOT}" \
    --episodes "${EPISODES}" \
    --replicate-id "${REPLICATE_ID}" \
    --logit-penalty "${LOGIT_PENALTY}" \
    --scenarios unannounced_meal_60g \
    --output "${RESULT_ROOT}/cpo_unannounced_meal_summary.json"

"${PYTHON_BIN}" - "${RESULT_ROOT}/cpo_unannounced_meal_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
payload = json.loads(summary_path.read_text())
missing = [
    item["condition"]
    for item in payload["summaries"]
    if item["delivered_unannounced_meals"] < 1
]
if missing:
    raise SystemExit(
        "Hidden meal was not actually delivered in: " + ", ".join(missing)
    )
print("Hidden-meal delivery in both arms: PASS")
PY

penalty_tag="${LOGIT_PENALTY%.*}"
NO_ROOT="${RESULT_ROOT}/unannounced_meal_60g/replicate${REPLICATE_ID}/no_shield/t1d/CPO"
SHIELD_ROOT="${RESULT_ROOT}/unannounced_meal_60g/replicate${REPLICATE_ID}/predictive_penalty${penalty_tag}/t1d/CPO"
SHIELD_METADATA="${SHIELD_ROOT}/child#002/seed0/with_shield_${penalty_tag}/run_metadata.txt"

grep -qx 'v1_use_meal_hyper_check=false' "${SHIELD_METADATA}"
"${PYTHON_BIN}" "${PAIR_VALIDATOR}" \
    "${NO_ROOT}" \
    "${SHIELD_ROOT}" \
    --expected-episodes "${EPISODES}" \
    --expected-cells 1 \
    --replicate-id "${REPLICATE_ID}" \
    --cohorts child \
    --patient-numbers 2 \
    --policy-seeds 0 \
    --logit-penalty "${LOGIT_PENALTY}"

echo "Local unannounced-meal comparison: PASS"
