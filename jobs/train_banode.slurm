#!/bin/bash
#SBATCH --job-name=banode-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --partition=gpu_course
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

EPOCHS="${1:-1}"
SEED="${2:-999}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
GLUCOALG_DIR="${PROJECT_DIR}/external/GlucoAlg"
VENV_DIR="${RTIDM_VENV:-$HOME/venvs/rtidm}"
MODEL_REL="saved_files/dynamics_predictor/ba_node_b5_p24_h24/${SEED}"

module purge
module load 2025
module load Python-bundle-PyPI/2025.04-GCCcore-14.2.0
module load CUDA/12.9.1

# Use the interpreter selected by the SURF module stack without allowing its
# central Python packages to shadow the project virtual environment.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Missing virtual environment: ${VENV_DIR}" >&2
    exit 1
fi

if [[ ! -d "${GLUCOALG_DIR}" ]]; then
    echo "Run sbatch from the repository root. Missing: ${GLUCOALG_DIR}" >&2
    exit 1
fi

if [[ -z "${TMPDIR:-}" || ! -d "${TMPDIR}" ]]; then
    echo "SLURM did not provide a usable TMPDIR." >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"

WORK_DIR="${TMPDIR}/GlucoAlg"
mkdir -p "${WORK_DIR}"
cp -a "${GLUCOALG_DIR}/." "${WORK_DIR}/"

sync_results() {
    if [[ -d "${WORK_DIR}/${MODEL_REL}" ]]; then
        mkdir -p "${GLUCOALG_DIR}/${MODEL_REL}"
        cp -a "${WORK_DIR}/${MODEL_REL}/." "${GLUCOALG_DIR}/${MODEL_REL}/"
        echo "Synced predictor outputs to ${GLUCOALG_DIR}/${MODEL_REL}"
    fi
}
trap sync_results EXIT
trap 'exit 143' TERM INT

cd "${WORK_DIR}"
echo "Training work directory: ${WORK_DIR}"

NPZ_COUNT="$(find saved_files/env_transitions -type f -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${NPZ_COUNT}" != "180" ]]; then
    echo "Expected 180 transition files, found ${NPZ_COUNT}." >&2
    exit 1
fi

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to train on CPU")
print(f"GPU: {torch.cuda.get_device_name(0)}")
properties = torch.cuda.get_device_properties(0)
print(f"GPU memory: {properties.total_memory / 2**30:.2f} GiB")
PY

echo "Training BA-NODE: epochs=${EPOCHS}, seed=${SEED}"
srun python -u 2.train_dynamics_predictor.py \
    --model ba_node \
    --epochs "${EPOCHS}" \
    --batch_size 32 \
    --history_length 24 \
    --prediction_length 24 \
    --n_basis 5 \
    --seed "${SEED}"
