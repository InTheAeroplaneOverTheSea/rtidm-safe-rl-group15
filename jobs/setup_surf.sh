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
GLUCOALG_PACKAGING_PATCH="${PROJECT_DIR}/patches/glucoalg_v1_setuptools_discovery.patch"

module purge
module load 2025
module load Python-bundle-PyPI/2025.04-GCCcore-14.2.0

unset PYTHONPATH
export PYTHONNOUSERSITE=1

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    python -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

test -n "${TMPDIR:-}"
test -d "${TMPDIR}"

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

# These editable installs provide dependencies and checkpoint/data locations.
# The author-v1 evaluation jobs themselves use freshly extracted, commit-pinned
# source archives via PYTHONPATH.
python -m pip install -e "${PROJECT_DIR}/external/GlucoSim"

# GlucoAlg commit 50d3134 predates setuptools' strict flat-layout package
# discovery. Patch only its packaging metadata during installation, then restore
# the exact pinned checkout so setup remains repeatable and the submodule clean.
GLUCOALG_PYPROJECT="${GLUCOALG_DIR}/pyproject.toml"
GLUCOALG_PYPROJECT_BACKUP=""
restore_glucoalg_pyproject() {
    if [[ -n "${GLUCOALG_PYPROJECT_BACKUP}" && -f "${GLUCOALG_PYPROJECT_BACKUP}" ]]; then
        cp "${GLUCOALG_PYPROJECT_BACKUP}" "${GLUCOALG_PYPROJECT}"
        rm -f "${GLUCOALG_PYPROJECT_BACKUP}"
        GLUCOALG_PYPROJECT_BACKUP=""
    fi
}

if ! grep -q '^\[tool\.setuptools\.packages\.find\]$' "${GLUCOALG_PYPROJECT}"; then
    test -f "${GLUCOALG_PACKAGING_PATCH}"
    GLUCOALG_PYPROJECT_BACKUP="$(mktemp)"
    cp "${GLUCOALG_PYPROJECT}" "${GLUCOALG_PYPROJECT_BACKUP}"
    trap restore_glucoalg_pyproject EXIT
    patch --batch --forward -p1 -d "${GLUCOALG_DIR}" < "${GLUCOALG_PACKAGING_PATCH}"
fi

python -m pip install -e "${GLUCOALG_DIR}"
restore_glucoalg_pyproject
trap - EXIT

python -u "${PROJECT_DIR}/scripts/download_author_v1_transitions.py" \
    --glucoalg-dir "${GLUCOALG_DIR}" \
    --download-dir "${TMPDIR}/author-v1-transitions-hf"

NPZ_COUNT="$(find "${GLUCOALG_DIR}/saved_files/env_transitions" -type f -name '*.npz' | wc -l | tr -d ' ')"
if [[ "${NPZ_COUNT}" != "180" ]]; then
    echo "Expected 180 transition files, found ${NPZ_COUNT}." >&2
    exit 1
fi

python -m pip check
python -m pip freeze > "${PROJECT_DIR}/surf_environment_freeze.txt"

echo "SURF evaluation environment is ready."
echo "Transition files: ${NPZ_COUNT}"
