#!/bin/bash
#SBATCH --job-name=v1-t1d-stress
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --partition=gpu_course
#SBATCH --time=01:00:00
#SBATCH --array=0-80%12
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

ALGORITHM="${1:?Usage: $0 ALGORITHM CONDITION STRESS_SCENARIO [episodes] [replicate_id] [penalty] [result_collection]}"
CONDITION="${2:?Usage: $0 ALGORITHM CONDITION STRESS_SCENARIO [episodes] [replicate_id] [penalty] [result_collection]}"
REQUESTED_STRESS_SCENARIO="${3:?Usage: $0 ALGORITHM CONDITION STRESS_SCENARIO [episodes] [replicate_id] [penalty] [result_collection]}"
case "${REQUESTED_STRESS_SCENARIO}" in
    sensor_sigma15) STRESS_SCENARIO="gaussian_sigma15" ;;
    *) STRESS_SCENARIO="${REQUESTED_STRESS_SCENARIO}" ;;
esac
EPISODES="${4:-1}"
REPLICATE_ID="${5:-0}"
LOGIT_PENALTY="${6:-10}"
RESULT_COLLECTION="${7:-author_v1_stress_evaluation}"
EVALUATION_SEED_BASE=22
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
VENV_DIR="${RTIDM_VENV:-$HOME/venvs/rtidm}"
GLUCOALG_ARCHIVE="${PROJECT_DIR}/author_v1_GlucoAlg_50d3134.tar.gz"
GLUCOSIM_ARCHIVE="${PROJECT_DIR}/author_v1_GlucoSim_6522d49.tar.gz"
GLUCOALG_ARCHIVE_SHA256="e2f775bd0a75fbbb657628708e98efe6d684dd5109edf7268c6e396cb5aa253f"
GLUCOSIM_ARCHIVE_SHA256="3e8f7e48e8606f7136bd372898fd855d84fc959a70736cdf4b8db34c88063ba0"
GLUCOSIM_COMPAT_PATCH="${PROJECT_DIR}/patches/glucosim_v1_python313.patch"
GLUCOSIM_STRESS_PATCH="${PROJECT_DIR}/patches/glucosim_v1_stress_diagnostics.patch"
GLUCOALG_STRESS_PATCH="${PROJECT_DIR}/patches/glucoalg_v1_stress_eval.patch"
STRESS_INTERFACE="${PROJECT_DIR}/interface_new.py"
STRESS_HOOK="${PROJECT_DIR}/scripts/author_v1_stress_hook.py"
AUTHOR_V1_REVISIONS="${PROJECT_DIR}/configs/author_v1_hf_revisions.json"
CHECKPOINT_VERIFIER="${PROJECT_DIR}/scripts/download_t1d_baseline_checkpoints.py"
TRANSITION_VERIFIER="${PROJECT_DIR}/scripts/download_author_v1_transitions.py"
POLICY_BASE="${PROJECT_DIR}/external/GlucoAlg/saved_models/t1d"
POLICY_MANIFEST="${PROJECT_DIR}/author_v1_all_baselines_evaluation/t1d_checkpoint_manifest.json"
TRANSITIONS_SOURCE="${PROJECT_DIR}/external/GlucoAlg/saved_files/env_transitions"
TRANSITION_MANIFEST="${PROJECT_DIR}/external/GlucoAlg/saved_files/author_v1_transition_manifest.json"
PREDICTOR_SOURCE="${PROJECT_DIR}/external/GlucoAlg/saved_files/dynamics_predictor/ba_node_b5_p24_h24/42"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?This job must be submitted as a Slurm array.}"

case "${ALGORITHM}" in
    CPO|CUP|FOCOPS|OnCRPO|PCPO|PPOLag|RCPO|TRPOLag) ;;
    *) echo "Unsupported algorithm: ${ALGORITHM}" >&2; exit 2 ;;
esac
case "${CONDITION}" in
    no_shield|predictive) ;;
    *) echo "Condition must be no_shield or predictive, got ${CONDITION}" >&2; exit 2 ;;
esac
case "${STRESS_SCENARIO}" in
    nominal|gaussian_sigma15|bias_positive|bias_negative|random_walk_drift|dropout_markov_v1|unannounced_meal_60g|drift_sigma0025_calibration|drift_sigma004_calibration) ;;
    *) echo "Unsupported stress scenario: ${STRESS_SCENARIO}" >&2; exit 2 ;;
esac
if [[ ! "${EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Episodes must be a positive integer, got ${EPISODES}" >&2
    exit 2
fi
if [[ ! "${REPLICATE_ID}" =~ ^[0-9]+$ ]]; then
    echo "Replicate id must be a nonnegative integer, got ${REPLICATE_ID}" >&2
    exit 2
fi
if [[ ! "${LOGIT_PENALTY}" =~ ^[0-9]+$ ]]; then
    echo "Logit penalty must be a nonnegative integer, got ${LOGIT_PENALTY}" >&2
    exit 2
fi
if [[ ! "${RESULT_COLLECTION}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Result collection must be a single safe directory name, got ${RESULT_COLLECTION}" >&2
    exit 2
fi
if [[ ! "${TASK_INDEX}" =~ ^[0-9]+$ ]] || (( TASK_INDEX > 80 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 80, got ${TASK_INDEX}" >&2
    exit 2
fi

COHORTS=(child adolescent adult)
COHORT_INDEX=$((TASK_INDEX / 27))
WITHIN_COHORT=$((TASK_INDEX % 27))
PATIENT_NUMBER=$((WITHIN_COHORT / 3 + 2))
POLICY_SEED=$((WITHIN_COHORT % 3))
COHORT="${COHORTS[COHORT_INDEX]}"
printf -v PATIENT_NAME '%s#%03d' "${COHORT}" "${PATIENT_NUMBER}"

module purge
module load 2025
module load Python-bundle-PyPI/2025.04-GCCcore-14.2.0

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

for required in \
    "${VENV_DIR}/bin/activate" \
    "${GLUCOALG_ARCHIVE}" \
    "${GLUCOSIM_ARCHIVE}" \
    "${GLUCOSIM_COMPAT_PATCH}" \
    "${GLUCOSIM_STRESS_PATCH}" \
    "${GLUCOALG_STRESS_PATCH}" \
    "${STRESS_INTERFACE}" \
    "${STRESS_HOOK}" \
    "${AUTHOR_V1_REVISIONS}" \
    "${CHECKPOINT_VERIFIER}" \
    "${POLICY_MANIFEST}"; do
    test -f "${required}"
done
test -n "${TMPDIR:-}"
test -d "${TMPDIR}"
printf '%s  %s\n%s  %s\n' \
    "${GLUCOALG_ARCHIVE_SHA256}" "${GLUCOALG_ARCHIVE}" \
    "${GLUCOSIM_ARCHIVE_SHA256}" "${GLUCOSIM_ARCHIVE}" \
    | sha256sum -c -

POLICY_SOURCE="${POLICY_BASE}/${COHORT}/${ALGORITHM}/seed${POLICY_SEED}"
test -f "${POLICY_SOURCE}/config.json"
test -f "${POLICY_SOURCE}/torch_save/epoch-2441.pt"
if [[ "${CONDITION}" == "predictive" ]]; then
    test -f "${PREDICTOR_SOURCE}/config.yaml"
    test -f "${PREDICTOR_SOURCE}/best_model.pth"
    test -f "${TRANSITION_VERIFIER}"
    test -f "${TRANSITION_MANIFEST}"
    test -d "${TRANSITIONS_SOURCE}"
fi
source "${VENV_DIR}/bin/activate"

python -u "${CHECKPOINT_VERIFIER}" \
    --glucoalg-dir "${PROJECT_DIR}/external/GlucoAlg" \
    --manifest "${POLICY_MANIFEST}" \
    --revisions-file "${AUTHOR_V1_REVISIONS}" \
    --algorithms "${ALGORITHM}" \
    --verify-only
if [[ "${CONDITION}" == "predictive" ]]; then
    python -u "${TRANSITION_VERIFIER}" \
        --glucoalg-dir "${PROJECT_DIR}/external/GlucoAlg" \
        --revisions-file "${AUTHOR_V1_REVISIONS}" \
        --verify-only
fi

WORK_ROOT="${TMPDIR}/author-v1-stress-${STRESS_SCENARIO}-r${REPLICATE_ID}-${ALGORITHM}-${SLURM_ARRAY_JOB_ID}-${TASK_INDEX}-${CONDITION}"
WORK_GLUCOALG="${WORK_ROOT}/GlucoAlg"
WORK_GLUCOSIM="${WORK_ROOT}/GlucoSim"
mkdir -p "${WORK_GLUCOALG}" "${WORK_GLUCOSIM}"
tar -xzf "${GLUCOALG_ARCHIVE}" -C "${WORK_GLUCOALG}"
tar -xzf "${GLUCOSIM_ARCHIVE}" -C "${WORK_GLUCOSIM}"
patch --batch --forward -p1 -d "${WORK_GLUCOSIM}" < "${GLUCOSIM_COMPAT_PATCH}"
patch --batch --forward -p1 -d "${WORK_GLUCOSIM}" < "${GLUCOSIM_STRESS_PATCH}"
cp "${STRESS_INTERFACE}" "${WORK_GLUCOALG}/interface_new.py"
cp "${STRESS_HOOK}" "${WORK_GLUCOALG}/author_v1_stress_hook.py"
patch --batch --forward -p1 -d "${WORK_GLUCOALG}" < "${GLUCOALG_STRESS_PATCH}"

# The January simulator uses its former glucobench package name internally.
ln -s "${WORK_GLUCOSIM}/glucosim" "${WORK_GLUCOSIM}/glucobench"

mkdir -p "${WORK_GLUCOALG}/saved_models/t1d/${COHORT}/${ALGORITHM}"
cp -a "${POLICY_SOURCE}" "${WORK_GLUCOALG}/saved_models/t1d/${COHORT}/${ALGORITHM}/"

if [[ "${CONDITION}" == "predictive" ]]; then
    PREDICTOR_TARGET="${WORK_GLUCOALG}/saved_files/dynamics_predictor/fe_b5_p24_h24/1"
    mkdir -p "$(dirname "${PREDICTOR_TARGET}")" "${WORK_GLUCOALG}/saved_files"
    cp -a "${PREDICTOR_SOURCE}" "${PREDICTOR_TARGET}"
    cp -a "${TRANSITIONS_SOURCE}" "${WORK_GLUCOALG}/saved_files/env_transitions"
fi

export PYTHONPATH="${WORK_GLUCOALG}:${WORK_GLUCOSIM}"

if [[ "${CONDITION}" == "no_shield" ]]; then
    SOURCE_TAG="no_shield"
    RESULT_CONDITION="no_shield"
else
    SOURCE_TAG="with_shield_${LOGIT_PENALTY%.*}"
    RESULT_CONDITION="predictive_penalty${LOGIT_PENALTY}"
fi

RESULT_REL="t1d/${ALGORITHM}/${PATIENT_NAME}/seed${POLICY_SEED}/${SOURCE_TAG}"
SOURCE_RESULT="${WORK_GLUCOALG}/diabetes_evaluation/${RESULT_REL}"
DEST_RESULT="${PROJECT_DIR}/${RESULT_COLLECTION}/${STRESS_SCENARIO}/replicate${REPLICATE_ID}/${RESULT_CONDITION}/${RESULT_REL}"

sync_results() {
    if [[ -d "${SOURCE_RESULT}" ]]; then
        mkdir -p "${DEST_RESULT}"
        cp -a "${SOURCE_RESULT}/." "${DEST_RESULT}/"
        {
            echo "code_stack=author_v1_stress"
            echo "glucoalg_commit=50d3134fd6f0e01019295be0d6b99f110f794753"
            echo "glucosim_commit=6522d49ee1e6c1fd34f287519a5d41df735bbd38"
            echo "glucosim_compat_patch=python313_typing_only"
            echo "glucosim_stress_diagnostics_patch_sha256=$(sha256sum "${GLUCOSIM_STRESS_PATCH}" | cut -d ' ' -f 1)"
            echo "stress_scenario=${STRESS_SCENARIO}"
            echo "requested_stress_scenario=${REQUESTED_STRESS_SCENARIO}"
            echo "stress_semantics=post_action_pre_step"
            echo "stress_interface_sha256=$(sha256sum "${STRESS_INTERFACE}" | cut -d ' ' -f 1)"
            echo "stress_hook_sha256=$(sha256sum "${STRESS_HOOK}" | cut -d ' ' -f 1)"
            echo "stress_eval_patch_sha256=$(sha256sum "${GLUCOALG_STRESS_PATCH}" | cut -d ' ' -f 1)"
            echo "algorithm=${ALGORITHM}"
            echo "condition=${CONDITION}"
            echo "result_collection=${RESULT_COLLECTION}"
            echo "device_mode=author_v1_cpu"
            echo "patient_name=${PATIENT_NAME}"
            echo "policy_seed=${POLICY_SEED}"
            echo "evaluation_seed=${EVALUATION_SEED_BASE}"
            echo "evaluation_seed_mode=explicit_key_before_every_episode"
            echo "replicate_id=${REPLICATE_ID}"
            echo "episodes=${EPISODES}"
            echo "checkpoint_epoch=2441"
            echo "checkpoint_revision_policy=author-v1-january-2026"
            if [[ "${CONDITION}" == "predictive" ]]; then
                echo "predictor_provenance=group_trained_ba_node_b5_p24_h24_seed42"
                echo "transition_dataset_revision=b9e59ff5f14fb36aacd69f9d57febf26e88de21d"
                echo "v1_use_meal_hyper_check=false"
                echo "v1_hyper_check_threshold_mgdl=250"
                echo "logit_penalty=${LOGIT_PENALTY}"
                sha256sum "${PREDICTOR_SOURCE}/best_model.pth"
            fi
            sha256sum "${POLICY_SOURCE}/torch_save/epoch-2441.pt"
        } > "${DEST_RESULT}/run_metadata.txt"
    fi
}
trap sync_results EXIT
trap 'exit 143' TERM INT

cd "${WORK_GLUCOALG}"
echo "Author v1 stress: scenario=${STRESS_SCENARIO}, replicate=${REPLICATE_ID}, algorithm=${ALGORITHM}, condition=${CONDITION}, cohort=${COHORT}, patient=${PATIENT_NAME}, policy_seed=${POLICY_SEED}, episodes=${EPISODES}, eval_seed_base=${EVALUATION_SEED_BASE}"

COMMON_ARGS=(
    t1d "${ALGORITHM}" "${PATIENT_NAME}" "${POLICY_SEED}"
    --epoch 2441
    --num-episodes "${EPISODES}"
    --stress-scenario "${STRESS_SCENARIO}"
    --evaluation-seed "${EVALUATION_SEED_BASE}"
    --replicate-id "${REPLICATE_ID}"
)
if [[ "${CONDITION}" == "no_shield" ]]; then
    srun python -u eval_run.py "${COMMON_ARGS[@]}" --shield-type none
else
    srun python -u eval_run.py "${COMMON_ARGS[@]}" \
        --shield \
        --shield-type predictive \
        --logit-penalty "${LOGIT_PENALTY}"
fi

test -f "${SOURCE_RESULT}/summary_results.csv"
test -f "${SOURCE_RESULT}/detailed_results.csv"
test -f "${SOURCE_RESULT}/stress_episode_metadata.json"
