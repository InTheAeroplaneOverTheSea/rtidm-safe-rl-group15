#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

restore_submodule_commit() {
    local submodule_path="$1"
    local expected_base="$2"
    local bundle_path="$3"
    local target_commit="$4"
    local current_head

    if [[ ! -d "${submodule_path}/.git" && ! -f "${submodule_path}/.git" ]]; then
        echo "Missing initialized submodule: ${submodule_path}" >&2
        echo "Run: git submodule update --init --recursive" >&2
        exit 1
    fi

    if ! git -C "${submodule_path}" diff --quiet \
        || ! git -C "${submodule_path}" diff --cached --quiet \
        || [[ -n "$(git -C "${submodule_path}" ls-files --others --exclude-standard)" ]]; then
        echo "Refusing to patch dirty submodule: ${submodule_path}" >&2
        exit 1
    fi

    current_head="$(git -C "${submodule_path}" rev-parse HEAD)"
    if [[ "${current_head}" != "${expected_base}" ]]; then
        echo "Unexpected base for ${submodule_path}" >&2
        echo "  expected: ${expected_base}" >&2
        echo "  actual:   ${current_head}" >&2
        exit 1
    fi

    git -C "${submodule_path}" fetch "${bundle_path}" "${target_commit}"
    git -C "${submodule_path}" checkout --detach "${target_commit}"
}

git -C "${PROJECT_DIR}" submodule update --init --recursive

restore_submodule_commit \
    "${PROJECT_DIR}/external/GlucoAlg" \
    "50d3134fd6f0e01019295be0d6b99f110f794753" \
    "${PROJECT_DIR}/vendor/glucoalg-paper-aligned.bundle" \
    "5cb78adccfccbe4f27f85316ed9d3b0a467def51"

restore_submodule_commit \
    "${PROJECT_DIR}/external/GlucoSim" \
    "ddd674e35a32a3a6406a0c2d5e7bf3c72f56dfec" \
    "${PROJECT_DIR}/vendor/glucosim-python-compat.bundle" \
    "c3c40a2986e79aa5e4af2cf47ecc085242d455be"

echo "Restored the reviewed group GlucoAlg and GlucoSim commits."
