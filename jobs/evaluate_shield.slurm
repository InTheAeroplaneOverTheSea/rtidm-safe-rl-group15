#!/bin/bash
#SBATCH --job-name=shield-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --partition=gpu_course
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

EPISODES="${1:-10}"
PERTURBATION="${2:-all}"
POLICY_SEED="${SLURM_ARRAY_TASK_ID:-${3:-0}}"
PATIENT_NAME="${4:-adolescent#001}"
HYPER_THRESHOLD="${5:-}"
LOGIT_PENALTY="${6:-10}"
EVAL_SEED="${7:-22}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
GLUCOALG_DIR="${PROJECT_DIR}/external/GlucoAlg"
VENV_DIR="${RTIDM_VENV:-$HOME/venvs/rtidm}"
DYNAMICS_REL="saved_files/dynamics_predictor/ba_node_b5_p24_h24/42"
RESULT_REL="diabetes_evaluation/t1d/CPO/${PATIENT_NAME}/seed${POLICY_SEED}"

module purge
module load 2025
module load Python-bundle-PyPI/2025.04-GCCcore-14.2.0
module load CUDA/12.9.1

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Missing virtual environment: ${VENV_DIR}" >&2
    exit 1
fi

if [[ ! -d "${GLUCOALG_DIR}" ]]; then
    echo "Run sbatch from the repository root. Missing: ${GLUCOALG_DIR}" >&2
    exit 1
fi

if [[ -z "${HYPER_THRESHOLD}" ]]; then
    echo "Missing calibrated predictive hyper threshold (<180 mg/dL)." >&2
    echo "Usage: sbatch jobs/evaluate_shield.slurm EPISODES PERTURBATION POLICY_SEED PATIENT HYPER_THRESHOLD [PENALTY] [EVAL_SEED]" >&2
    exit 1
fi

python - "${HYPER_THRESHOLD}" "${LOGIT_PENALTY}" <<'PY'
import sys

threshold = float(sys.argv[1])
penalty = float(sys.argv[2])
if not 0.0 < threshold < 180.0:
    raise SystemExit("HYPER_THRESHOLD must be between 0 and 180 mg/dL.")
if penalty not in {5.0, 10.0, 15.0}:
    raise SystemExit("PENALTY must be one of 5, 10, or 15.")
PY

if [[ "${PERTURBATION}" == "all" ]]; then
    PERTURBATIONS=(none cgm_bias_plus30)
else
    PERTURBATIONS=("${PERTURBATION}")
fi
for perturbation_name in "${PERTURBATIONS[@]}"; do
    perturbation_path="${PROJECT_DIR}/configs/perturbations/${perturbation_name}.yaml"
    if [[ ! -f "${perturbation_path}" ]]; then
        echo "Unknown perturbation config: ${perturbation_path}" >&2
        exit 1
    fi
done

if [[ -z "${TMPDIR:-}" || ! -d "${TMPDIR}" ]]; then
    echo "SLURM did not provide a usable TMPDIR." >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"

python - <<'PY'
import importlib.util
import torch

required = ["glucosim", "gymnasium", "jax", "matplotlib", "pandas"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing evaluation dependencies: " + ", ".join(missing)
        + ". Re-submit jobs/setup_surf.slurm before evaluation."
    )
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing predictive-shield evaluation on CPU")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

WORK_ROOT="${TMPDIR}/rtidm-safe-rl-group15"
WORK_GLUCOALG="${WORK_ROOT}/external/GlucoAlg"
mkdir -p "${WORK_ROOT}/external"
cp -a "${PROJECT_DIR}/experiments" "${WORK_ROOT}/"
cp -a "${PROJECT_DIR}/configs" "${WORK_ROOT}/"
cp -a "${PROJECT_DIR}/src" "${WORK_ROOT}/"
cp -a "${GLUCOALG_DIR}" "${WORK_ROOT}/external/"

sync_results() {
    local source_dir="${WORK_GLUCOALG}/${RESULT_REL}"
    local destination_dir="${GLUCOALG_DIR}/${RESULT_REL}"
    if [[ -d "${source_dir}" ]]; then
        mkdir -p "${destination_dir}"
        cp -a "${source_dir}/." "${destination_dir}/"
        echo "Synced evaluation outputs to ${destination_dir}"
    fi
}
trap sync_results EXIT
trap 'exit 143' TERM INT

NPZ_COUNT="$(find "${WORK_GLUCOALG}/saved_files/env_transitions" -type f -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${NPZ_COUNT}" != "180" ]]; then
    echo "Expected 180 transition files, found ${NPZ_COUNT}." >&2
    exit 1
fi

POLICY_ROOT="${WORK_GLUCOALG}/saved_models/t1d/adolescent/CPO/seed${POLICY_SEED}"
if [[ ! -f "${POLICY_ROOT}/config.json" || ! -f "${POLICY_ROOT}/torch_save/epoch-2441.pt" ]]; then
    echo "Missing CPO seed ${POLICY_SEED} checkpoint under ${POLICY_ROOT}." >&2
    exit 1
fi

if [[ ! -f "${WORK_GLUCOALG}/${DYNAMICS_REL}/config.yaml" || ! -f "${WORK_GLUCOALG}/${DYNAMICS_REL}/best_model.pth" ]]; then
    echo "Missing seed-42 BA-NODE best checkpoint under ${WORK_GLUCOALG}/${DYNAMICS_REL}." >&2
    exit 1
fi

cd "${WORK_ROOT}"
echo "Evaluating CPO seed=${POLICY_SEED}, patient=${PATIENT_NAME}, episodes=${EPISODES}, perturbation=${PERTURBATION}, hyper_threshold=${HYPER_THRESHOLD}, penalty=${LOGIT_PENALTY}, eval_seed=${EVAL_SEED}"

for perturbation_name in "${PERTURBATIONS[@]}"; do
    perturbation_rel="configs/perturbations/${perturbation_name}.yaml"

    echo "Condition: perturbation=${perturbation_name}, shield=none"
    srun python -u experiments/evaluate_with_perturbations.py \
        t1d CPO "${PATIENT_NAME}" "${POLICY_SEED}" \
        --epoch 2441 \
        --num-episodes "${EPISODES}" \
        --shield-type none \
        --eval-seed "${EVAL_SEED}" \
        --perturbation-config "${perturbation_rel}"

    echo "Condition: perturbation=${perturbation_name}, shield=predictive"
    srun python -u experiments/evaluate_with_perturbations.py \
        t1d CPO "${PATIENT_NAME}" "${POLICY_SEED}" \
        --epoch 2441 \
        --num-episodes "${EPISODES}" \
        --shield \
        --shield-type predictive \
        --dynamics-model-dir "${DYNAMICS_REL}" \
        --hyper-check-threshold "${HYPER_THRESHOLD}" \
        --logit-penalty "${LOGIT_PENALTY}" \
        --eval-seed "${EVAL_SEED}" \
        --perturbation-config "${perturbation_rel}"
done

echo "Evaluation summaries:"
find "${WORK_GLUCOALG}/${RESULT_REL}" -mindepth 2 -maxdepth 2 -name summary_results.csv -print -exec sed -n '1,20p' {} \;
