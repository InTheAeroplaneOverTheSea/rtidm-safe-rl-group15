#!/bin/bash

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing to run setup on a login node." >&2
    echo "Submit it with: sbatch jobs/setup_surf.slurm" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${RTIDM_VENV:-$HOME/venvs/rtidm}"
GLUCOALG_DIR="${PROJECT_DIR}/external/GlucoAlg"

module purge
module load 2025
module load Python-bundle-PyPI/2025.04-GCCcore-14.2.0
module load CUDA/12.9.1

# Python-bundle-PyPI exposes SURF's central site-packages through PYTHONPATH.
# Keep its Python interpreter and CUDA toolchain, but prevent those packages
# from shadowing the versions installed in this project's virtual environment.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    python -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python - <<'PY'
import sys

central_site_packages = [
    path
    for path in sys.path
    if "/gpfs/admin/_hpc/sw/" in path and "site-packages" in path
]
if central_site_packages:
    raise SystemExit(
        "SURF central site-packages are still visible inside the virtual "
        f"environment: {central_site_packages}"
    )
print("Virtual environment isolation check passed.")
PY

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
    torch \
    einops \
    numpy \
    tensorboard \
    stable-baselines3 \
    tqdm \
    pyyaml \
    huggingface_hub

# Evaluation imports GlucoSim directly and needs its JAX/Gymnasium stack.
python -m pip install -e "${PROJECT_DIR}/external/GlucoSim"

# Importing the bundled OmniSafe package eagerly loads its algorithm modules.
# Install GlucoAlg's declared runtime dependencies (including rich and wandb)
# instead of installing a separate PyPI copy of OmniSafe.
python -m pip install -e "${GLUCOALG_DIR}"

python - "${GLUCOALG_DIR}/saved_files" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="safe-diabetes-benchmark/transition_datasets",
    repo_type="dataset",
    local_dir=sys.argv[1],
    allow_patterns=["env_transitions/**"],
)
PY

NPZ_COUNT="$(find "${GLUCOALG_DIR}/saved_files/env_transitions" -type f -name '*.npz' | wc -l | tr -d ' ')"
if [[ "${NPZ_COUNT}" != "180" ]]; then
    echo "Expected 180 transition files, found ${NPZ_COUNT}." >&2
    exit 1
fi

python -m pip check
python -m pip freeze > "${PROJECT_DIR}/surf_environment_freeze.txt"
python - "${GLUCOALG_DIR}" <<'PY'
import sys

# GlucoAlg uses repository-root imports such as ``import dynamics_trainers``.
# Match the evaluation wrapper by placing that root on sys.path for this check.
sys.path.insert(0, sys.argv[1])

import einops
import glucosim
import glucosim.diabetes_cmdp
import gymnasium
import jax
import matplotlib
import numpy
import pandas
import stable_baselines3
import torch
import yaml

print(f"PyTorch: {torch.__version__}")
print(f"JAX: {jax.__version__}")
print(f"CUDA visible on staging node: {torch.cuda.is_available()} (False is normal here)")
print("SURF training environment is ready.")
PY

echo "Transition files: ${NPZ_COUNT}"
