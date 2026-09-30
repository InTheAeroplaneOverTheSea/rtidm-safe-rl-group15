#!/bin/bash
#SBATCH --job-name=t1d-ckpts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=staging
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
VENV_DIR="${RTIDM_VENV:-$HOME/venvs/rtidm}"
GLUCOALG_DIR="${PROJECT_DIR}/external/GlucoAlg"
MANIFEST="${PROJECT_DIR}/author_v1_all_baselines_evaluation/t1d_checkpoint_manifest.json"

module purge
module load 2025
module load Python-bundle-PyPI/2025.04-GCCcore-14.2.0

unset PYTHONPATH
export PYTHONNOUSERSITE=1

test -f "${VENV_DIR}/bin/activate"
test -n "${TMPDIR:-}"
test -d "${TMPDIR}"

source "${VENV_DIR}/bin/activate"

algorithms=("$@")
if (( ${#algorithms[@]} == 0 )); then
    algorithms=(CPO CUP FOCOPS OnCRPO PCPO PPOLag RCPO TRPOLag)
fi

python -u "${PROJECT_DIR}/scripts/download_t1d_baseline_checkpoints.py" \
    --glucoalg-dir "${GLUCOALG_DIR}" \
    --download-dir "${TMPDIR}/t1d-baselines-hf" \
    --manifest "${MANIFEST}" \
    --algorithms "${algorithms[@]}"

for algorithm in "${algorithms[@]}"; do
    for cohort in child adolescent adult; do
        for seed in 0 1 2; do
            root="${GLUCOALG_DIR}/saved_models/t1d/${cohort}/${algorithm}/seed${seed}"
            test -f "${root}/config.json"
            test -f "${root}/torch_save/epoch-2441.pt"
        done
    done
done

echo "Installed and verified T1D policy checkpoints for: ${algorithms[*]}"
