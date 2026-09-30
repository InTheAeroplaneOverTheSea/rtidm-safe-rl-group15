#!/bin/bash
#SBATCH --job-name=v1-sources
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=staging
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
GLUCOALG_COMMIT="50d3134fd6f0e01019295be0d6b99f110f794753"
GLUCOSIM_COMMIT="6522d49ee1e6c1fd34f287519a5d41df735bbd38"

test -n "${TMPDIR:-}"
test -d "${TMPDIR}"

git clone --quiet https://github.com/safe-autonomy-lab/GlucoAlg.git "${TMPDIR}/GlucoAlg"
git clone --quiet https://github.com/safe-autonomy-lab/GlucoSim.git "${TMPDIR}/GlucoSim"

git -C "${TMPDIR}/GlucoAlg" archive \
    --format=tar.gz \
    --output="${PROJECT_DIR}/author_v1_GlucoAlg_50d3134.tar.gz" \
    "${GLUCOALG_COMMIT}"
git -C "${TMPDIR}/GlucoSim" archive \
    --format=tar.gz \
    --output="${PROJECT_DIR}/author_v1_GlucoSim_6522d49.tar.gz" \
    "${GLUCOSIM_COMMIT}"

sha256sum \
    "${PROJECT_DIR}/author_v1_GlucoAlg_50d3134.tar.gz" \
    "${PROJECT_DIR}/author_v1_GlucoSim_6522d49.tar.gz"
